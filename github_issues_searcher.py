#!/usr/bin/env python3
"""
GitHub Issues Searcher

Provides semantic search functionality for GitHub issues stored in the database.
"""

import sys
import argparse
from typing import List, Dict, Any, Optional
from config import Config
from database_manager import DatabaseManager
from embedding_model import create_embedding_model
from cross_encoder import create_reranker

class GitHubIssuesSearcher:
    """Searches GitHub issues using semantic similarity."""
    
    def __init__(self, db_manager: DatabaseManager, embedding_model=None, reranker=None):
        self.db_manager = db_manager
        self.embedding_model = embedding_model
        self.reranker = reranker
    
    def search_issues_semantic(self, query: str, limit: int = 10, 
                              repository: Optional[str] = None,
                              state: Optional[str] = None,
                              author: Optional[str] = None,
                              labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search GitHub issues using semantic similarity.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            repository: Optional repository filter
            state: Optional state filter (open, closed, all)
            author: Optional author filter
            labels: Optional list of label filters
            
        Returns:
            List of matching issues with similarity scores
        """
        if not self.embedding_model:
            print("Warning: No embedding model available for semantic search")
            return []
        
        try:
            # Compute query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search issues
            results = self.db_manager.search_github_issues(
                query=query,
                query_embedding=query_embedding,
                limit=limit,
                repository=repository,
                state=state,
                author=author,
                labels=labels
            )
            
            # Add search metadata
            for result in results:
                result['search_type'] = 'semantic'
                result['document_type'] = 'github_issue'
            
            # Apply cross-encoder reranking if available
            if self.reranker and results:
                print(f"🔄 Applying cross-encoder reranking to GitHub issues...")
                results = self.reranker.rerank_results(query, results, top_k=limit)
                print(f"✅ GitHub issues reranking complete!")
            
            return results
            
        except Exception as e:
            print(f"Error in GitHub issues semantic search: {e}")
            return []
    
    def search_issues_sql(self, query: str, limit: int = 10,
                          repository: Optional[str] = None,
                          state: Optional[str] = None,
                          author: Optional[str] = None,
                          labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search GitHub issues using SQL LIKE queries.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            repository: Optional repository filter
            state: Optional state filter
            author: Optional author filter
            labels: Optional list of label filters
            
        Returns:
            List of matching issues
        """
        try:
            results = self.db_manager.search_github_issues_sql(
                query=query,
                limit=limit,
                repository=repository,
                state=state,
                author=author,
                labels=labels
            )
            
            # Add search metadata
            for result in results:
                result['search_type'] = 'sql'
                result['document_type'] = 'github_issue'
            
            return results
            
        except Exception as e:
            print(f"Error in GitHub issues SQL search: {e}")
            return []
    
    def search_issues_hybrid(self, query: str, search_type: str = 'both', 
                            limit: int = 10, repository: Optional[str] = None,
                            state: Optional[str] = None, author: Optional[str] = None,
                            labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Hybrid search combining SQL and semantic search.
        
        Args:
            query: Search query text
            search_type: 'sql', 'semantic', or 'both'
            limit: Maximum results per search type
            repository: Optional repository filter
            state: Optional state filter
            author: Optional author filter
            labels: Optional list of label filters
            
        Returns:
            Dictionary with search results and metadata
        """
        results = {
            'sql_results': [],
            'semantic_results': [],
            'combined_results': [],
            'search_metadata': {
                'search_term': query,
                'search_type': search_type,
                'repository': repository,
                'state': state,
                'author': author,
                'labels': labels,
                'total_results': 0,
                'sql_count': 0,
                'semantic_count': 0
            }
        }
        
        # SQL search
        if search_type in ['sql', 'both']:
            sql_results = self.search_issues_sql(
                query, limit, repository, state, author, labels
            )
            results['sql_results'] = sql_results
            results['search_metadata']['sql_count'] = len(sql_results)
        
        # Semantic search
        if search_type in ['semantic', 'both'] and self.embedding_model:
            semantic_results = self.search_issues_semantic(
                query, limit, repository, state, author, labels
            )
            results['semantic_results'] = semantic_results
            results['search_metadata']['semantic_count'] = len(semantic_results)
        
        # Combine results
        if search_type == 'both':
            combined = []
            seen_ids = set()
            
            # Add semantic results first (already sorted by relevance)
            for issue in results.get('semantic_results', []):
                if issue['issue_id'] not in seen_ids:
                    issue_copy = dict(issue)
                    if issue['issue_id'] in [r['issue_id'] for r in results.get('sql_results', [])]:
                        issue_copy['search_type'] = 'both'
                    combined.append(issue_copy)
                    seen_ids.add(issue['issue_id'])
            
            # Add SQL results (avoiding duplicates)
            for issue in results.get('sql_results', []):
                if issue['issue_id'] not in seen_ids:
                    issue_copy = dict(issue)
                    combined.append(issue_copy)
                    seen_ids.add(issue['issue_id'])
            
            # Sort by relevance (both first, then semantic, then sql)
            combined.sort(key=lambda x: (
                x.get('search_type') == 'both',
                x.get('search_type') == 'semantic',
                x.get('similarity_score', 0) if x.get('search_type') == 'semantic' else 0
            ), reverse=True)
            
            results['combined_results'] = combined[:limit]
            results['search_metadata']['total_results'] = len(results['combined_results'])
        else:
            # Single search type
            if search_type == 'sql':
                results['combined_results'] = results['sql_results']
                results['search_metadata']['total_results'] = len(results['sql_results'])
            else:  # semantic
                results['combined_results'] = results['semantic_results']
                results['search_metadata']['total_results'] = len(results['semantic_results'])
        
        return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Search GitHub issues using semantic similarity')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--type', choices=['sql', 'semantic', 'both'], default='both',
                       help='Search type (default: both)')
    parser.add_argument('--limit', type=int, default=10, help='Maximum results (default: 10)')
    parser.add_argument('--repository', help='Filter by repository (e.g., microsoft/vscode)')
    parser.add_argument('--state', choices=['open', 'closed', 'all'], default='all',
                       help='Filter by issue state (default: all)')
    parser.add_argument('--author', help='Filter by author username')
    parser.add_argument('--labels', nargs='+', help='Filter by labels')
    parser.add_argument('--show-sql', action='store_true', help='Show SQL queries being executed')
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = Config()
        db_config = config.get_db_config()
        
        if not all(db_config.get(key) for key in ['host', 'database', 'user', 'password']:
            print("❌ Incomplete database configuration")
            return 1
        
        print(f"✅ Database config loaded")
        
        # Initialize database manager
        db_manager = DatabaseManager(db_config)
        if not db_manager.connect():
            print("❌ Failed to connect to database")
            return 1
        
        print("✅ Database connection established")
        
        # Load embedding model and reranker
        embedding_model = None
        reranker = None
        
        if args.type in ['semantic', 'both']:
            embedding_model = create_embedding_model()
            if not embedding_model:
                print("Warning: Could not load embedding model")
                if args.type == 'semantic':
                    args.type = 'sql'
                elif args.type == 'both':
                    args.type = 'sql'
        
        # Load reranker for better result relevance
        reranker = create_reranker()
        
        # Create searcher
        searcher = GitHubIssuesSearcher(db_manager, embedding_model, reranker)
        
        # Perform search
        print(f"🔍 Searching GitHub issues for: '{args.query}'")
        print(f"Type: {args.type}")
        if args.repository:
            print(f"Repository: {args.repository}")
        if args.state:
            print(f"State: {args.state}")
        if args.author:
            print(f"Author: {args.author}")
        if args.labels:
            print(f"Labels: {args.labels}")
        print()
        
        # Perform search
        results = searcher.search_issues_hybrid(
            query=args.query,
            search_type=args.type,
            limit=args.limit,
            repository=args.repository,
            state=args.state,
            author=args.author,
            labels=args.labels
        )
        
        if not results['combined_results']:
            print("No issues found")
            return 0
        
        # Display results
        metadata = results['search_metadata']
        print(f"Found {metadata['total_results']} issues:")
        print(f"  SQL: {metadata.get('sql_count', 0)}")
        print(f"  Semantic: {metadata.get('semantic_count', 0)}")
        print()
        
        # Show results
        for i, issue in enumerate(results['combined_results'], 1):
            print(f"{i}. {issue.get('title', 'No title')}")
            print(f"   Repository: {issue.get('repository', 'Unknown')}")
            print(f"   Author: {issue.get('author', 'Unknown')}")
            print(f"   State: {issue.get('state', 'Unknown')}")
            print(f"   Number: #{issue.get('issue_number', 'Unknown')}")
            print(f"   Labels: {', '.join(issue.get('labels', [])) or 'None'}")
            print(f"   Assignees: {', '.join(issue.get('assignees', [])) or 'None'}")
            print(f"   Created: {issue.get('created_at', 'Unknown')}")
            print(f"   Updated: {issue.get('updated_at', 'Unknown')}")
            print(f"   Comments: {issue.get('comments_count', 0)}")
            print(f"   URL: {issue.get('html_url', 'Unknown')}")
            
            if 'similarity_score' in issue:
                print(f"   Similarity: {issue['similarity_score']:.3f}")
            
            if 'cross_encoder_score' in issue:
                print(f"   Cross-Encoder: {issue['cross_encoder_score']:.3f}")
            
            search_type = issue.get('search_type', 'unknown')
            print(f"   Found via: {search_type}")
            
            # Show snippet of body
            body = issue.get('body', '')
            if body:
                snippet = body[:200] + "..." if len(body) > 200 else body
                print(f"   Body: {snippet}")
            print()
        
        # Clean up
        db_manager.disconnect()
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
