#!/usr/bin/env python3
"""
GitHub Issues Fetcher and Indexer

Fetches issues from GitHub repositories and stores them in the database
with semantic embeddings for search.
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from config import Config
from database_manager import DatabaseManager

@dataclass
class GitHubIssue:
    """Represents a GitHub issue."""
    issue_id: int
    repository: str
    title: str
    body: str
    state: str
    labels: List[str]
    assignees: List[str]
    author: str
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime]
    comments_count: int
    html_url: str
    number: int
    milestone: Optional[str]
    reactions: Dict[str, int]

class GitHubIssuesFetcher:
    """Fetches and indexes GitHub issues."""
    
    def __init__(self, config: Config):
        self.config = config
        self.github_token = config.get_github_token()
        self.db_manager = None
        self.session = requests.Session()
        
        if self.github_token:
            self.session.headers.update({
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            })
        
        # Rate limiting
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = 0
    
    def connect_database(self) -> bool:
        """Connect to the database."""
        db_config = self.config.get_db_config()
        if not all(db_config.get(key) for key in ['host', 'database', 'user', 'password']):
            print("❌ Incomplete database configuration")
            return False
        
        self.db_manager = DatabaseManager(db_config)
        return self.db_manager.connect()
    
    def check_rate_limit(self):
        """Check and respect GitHub rate limits."""
        if self.rate_limit_remaining <= 10:
            if self.rate_limit_reset > time.time():
                wait_time = self.rate_limit_reset - time.time() + 60
                print(f"⚠️  Rate limit reached. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
            else:
                # Reset rate limit
                self.rate_limit_remaining = 5000
    
    def fetch_repository_issues(self, repo: str, state: str = 'all', per_page: int = 100) -> List[GitHubIssue]:
        """Fetch all issues from a specific repository."""
        print(f"🔍 Fetching issues from {repo} (state: {state})")
        
        issues = []
        page = 1
        
        while True:
            self.check_rate_limit()
            
            url = f"https://api.github.com/repos/{repo}/issues"
            params = {
                'state': state,
                'per_page': per_page,
                'page': page,
                'sort': 'updated',
                'direction': 'desc'
            }
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 5000))
                self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
                
                page_issues = response.json()
                
                if not page_issues:
                    break
                
                for issue_data in page_issues:
                    # Skip pull requests (they have pull_request field)
                    if 'pull_request' in issue_data:
                        continue
                    
                    issue = GitHubIssue(
                        issue_id=issue_data['id'],
                        repository=repo,
                        title=issue_data['title'],
                        body=issue_data['body'] or '',
                        state=issue_data['state'],
                        labels=[label['name'] for label in issue_data.get('labels', [])],
                        assignees=[assignee['login'] for assignee in issue_data.get('assignees', [])],
                        author=issue_data['user']['login'],
                        created_at=datetime.fromisoformat(issue_data['created_at'].replace('Z', '+00:00')),
                        updated_at=datetime.fromisoformat(issue_data['updated_at'].replace('Z', '+00:00')),
                        closed_at=datetime.fromisoformat(issue_data['closed_at'].replace('Z', '+00:00')) if issue_data['closed_at'] else None,
                        comments_count=issue_data['comments'],
                        html_url=issue_data['html_url'],
                        number=issue_data['number'],
                        milestone=issue_data['milestone']['title'] if issue_data['milestone'] else None,
                        reactions=issue_data.get('reactions', {})
                    )
                    issues.append(issue)
                
                print(f"   📄 Page {page}: {len(page_issues)} issues")
                page += 1
                
                # Small delay to be respectful
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error fetching issues from {repo}: {e}")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                break
        
        print(f"✅ Fetched {len(issues)} issues from {repo}")
        return issues
    
    def store_issues_in_database(self, issues: List[GitHubIssue]) -> bool:
        """Store issues in the database."""
        if not self.db_manager:
            print("❌ Database not connected")
            return False
        
        try:
            # Create tables if they don't exist
            self.create_github_tables()
            
            stored_count = 0
            for issue in issues:
                if self.store_single_issue(issue):
                    stored_count += 1
            
            print(f"✅ Stored {stored_count}/{len(issues)} issues in database")
            return stored_count > 0
            
        except Exception as e:
            print(f"❌ Error storing issues: {e}")
            return False
    
    def create_github_tables(self):
        """Create GitHub issues tables if they don't exist."""
        with self.db_manager.connection.cursor() as cursor:
            # GitHub issues table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS github_issues (
                    id SERIAL PRIMARY KEY,
                    issue_id BIGINT UNIQUE NOT NULL,
                    repository VARCHAR(255) NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT,
                    state VARCHAR(50) NOT NULL,
                    labels TEXT[], -- Array of label names
                    assignees TEXT[], -- Array of assignee usernames
                    author VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    closed_at TIMESTAMP WITH TIME ZONE,
                    comments_count INTEGER DEFAULT 0,
                    html_url TEXT NOT NULL,
                    issue_number INTEGER NOT NULL,
                    milestone VARCHAR(255),
                    reactions JSONB,
                    created_at_db TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # GitHub issue embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS github_issue_embeddings (
                    id SERIAL PRIMARY KEY,
                    issue_id BIGINT NOT NULL REFERENCES github_issues(issue_id) ON DELETE CASCADE,
                    embedding_model VARCHAR(100) NOT NULL,
                    embedding_vector vector(768), -- e5-base model dimension
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(issue_id, embedding_model)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_github_issues_repository ON github_issues(repository)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_github_issues_state ON github_issues(state)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_github_issues_author ON github_issues(author)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_github_issues_created_at ON github_issues(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_github_issues_labels ON github_issues USING GIN(labels)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_github_issues_assignees ON github_issues USING GIN(assignees)")
            
            self.db_manager.connection.commit()
            print("✅ GitHub tables created/verified")
    
    def store_single_issue(self, issue: GitHubIssue) -> bool:
        """Store a single issue in the database."""
        try:
            with self.db_manager.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO github_issues (
                        issue_id, repository, title, body, state, labels, assignees,
                        author, created_at, updated_at, closed_at, comments_count,
                        html_url, issue_number, milestone, reactions
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (issue_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        body = EXCLUDED.body,
                        state = EXCLUDED.state,
                        labels = EXCLUDED.labels,
                        assignees = EXCLUDED.assignees,
                        updated_at = EXCLUDED.updated_at,
                        closed_at = EXCLUDED.closed_at,
                        comments_count = EXCLUDED.comments_count,
                        milestone = EXCLUDED.milestone,
                        reactions = EXCLUDED.reactions
                """, (
                    issue.issue_id, issue.repository, issue.title, issue.body,
                    issue.state, issue.labels, issue.assignees, issue.author,
                    issue.created_at, issue.updated_at, issue.closed_at,
                    issue.comments_count, issue.html_url, issue.number,
                    issue.milestone, json.dumps(issue.reactions)
                ))
                
                return True
                
        except Exception as e:
            print(f"❌ Error storing issue {issue.issue_id}: {e}")
            return False
    
    def fetch_and_store_repositories(self, repositories: List[str], state: str = 'all') -> bool:
        """Fetch and store issues from multiple repositories."""
        print(f"🚀 Starting GitHub issues fetch for {len(repositories)} repositories")
        print("=" * 60)
        
        if not self.connect_database():
            return False
        
        total_issues = 0
        successful_repos = 0
        
        for repo in repositories:
            try:
                issues = self.fetch_repository_issues(repo, state)
                if issues:
                    if self.store_issues_in_database(issues):
                        total_issues += len(issues)
                        successful_repos += 1
                
                # Be respectful to GitHub API
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Failed to process repository {repo}: {e}")
        
        print("=" * 60)
        print(f"✅ Completed! Successfully processed {successful_repos}/{len(repositories)} repositories")
        print(f"📊 Total issues stored: {total_issues}")
        
        return successful_repos > 0

def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python github_issues_fetcher.py <repo1> [repo2] [repo3] ...")
        print("Example: python github_issues_fetcher.py 'microsoft/vscode' 'github/copilot'")
        return 1
    
    try:
        config = Config()
        fetcher = GitHubIssuesFetcher(config)
        
        repositories = sys.argv[1:]
        print(f"📋 Repositories to fetch: {repositories}")
        
        success = fetcher.fetch_and_store_repositories(repositories)
        
        if fetcher.db_manager:
            fetcher.db_manager.disconnect()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
