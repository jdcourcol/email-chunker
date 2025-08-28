# GitHub Issues Semantic Search

This system provides semantic search functionality for GitHub repository issues, allowing you to search through issues using natural language queries and find the most relevant results.

## Features

✅ **Semantic Search**: Find issues using natural language queries  
✅ **Hybrid Search**: Combine semantic and SQL search for better results  
✅ **Cross-Encoder Reranking**: Improve result relevance with advanced reranking  
✅ **Rich Filtering**: Filter by repository, state, author, labels, and more  
✅ **Rate Limiting**: Respects GitHub API rate limits  
✅ **Batch Processing**: Fetch issues from multiple repositories efficiently  

## Architecture

The system consists of several components:

1. **GitHub Issues Fetcher** (`github_issues_fetcher.py`): Fetches issues from GitHub repositories
2. **Database Storage**: Stores issues and their embeddings in PostgreSQL with pgvector
3. **Embedding Computation** (`compute_github_embeddings.py`): Computes semantic embeddings for issues
4. **Semantic Search** (`github_issues_searcher.py`): Provides search functionality
5. **Database Manager**: Handles database operations and pgvector similarity search

## Setup

### 1. Prerequisites

- PostgreSQL database with pgvector extension
- Python 3.8+
- GitHub Personal Access Token (optional, for higher rate limits)

### 2. Configuration

Add your GitHub token to your `config.json` file:

```json
{
  "github_token": "ghp_your_token_here",
  "db_host": "localhost",
  "db_name": "your_database",
  "db_user": "your_user",
  "db_password": "your_password"
}
```

Or set the `GITHUB_TOKEN` environment variable:

```bash
export GITHUB_TOKEN="ghp_your_token_here"
```

### 3. Install Dependencies

```bash
uv sync
```

## Usage

### 1. Fetch Issues from Repositories

Fetch issues from one or more GitHub repositories:

```bash
# Fetch from a single repository
uv run github_issues_fetcher.py "microsoft/vscode"

# Fetch from multiple repositories
uv run github_issues_fetcher.py "microsoft/vscode" "github/copilot" "openai/whisper"

# Fetch only open issues
uv run github_issues_fetcher.py "microsoft/vscode" --state open
```

**Note**: The fetcher automatically:
- Skips pull requests (only fetches actual issues)
- Respects GitHub API rate limits
- Updates existing issues if they've changed
- Creates necessary database tables

### 2. Compute Embeddings

After fetching issues, compute semantic embeddings:

```bash
uv run compute_github_embeddings.py
```

This will:
- Find all issues without embeddings
- Compute embeddings using the configured model (default: `intfloat/e5-base`)
- Store embeddings in the database for fast similarity search

### 3. Search Issues

Search through issues using semantic similarity:

```bash
# Basic semantic search
uv run github_issues_searcher.py "performance optimization"

# Search with filters
uv run github_issues_searcher.py "bug fix" --repository "microsoft/vscode" --state open

# Search by author
uv run github_issues_searcher.py "documentation" --author "octocat"

# Search by labels
uv run github_issues_searcher.py "feature request" --labels "enhancement" "good first issue"

# Limit results
uv run github_issues_searcher.py "security" --limit 5

# Search type options
uv run github_issues_searcher.py "query" --type semantic    # Semantic only
uv run github_issues_searcher.py "query" --type sql         # SQL only  
uv run github_issues_searcher.py "query" --type both        # Both (default)
```

## Search Options

### Search Types

- **`--type semantic`**: Uses pgvector similarity search only
- **`--type sql`**: Uses SQL LIKE queries only
- **`--type both`**: Combines both approaches (default)

### Filters

- **`--repository`**: Filter by specific repository (e.g., "microsoft/vscode")
- **`--state`**: Filter by issue state (`open`, `closed`, `all`)
- **`--author`**: Filter by author username
- **`--labels`**: Filter by one or more labels
- **`--limit`**: Maximum number of results to return

## Database Schema

The system creates two main tables:

### `github_issues`

Stores the main issue data:
- `issue_id`: GitHub's unique issue ID
- `repository`: Repository name (e.g., "microsoft/vscode")
- `title`: Issue title
- `body`: Issue description
- `state`: Issue state (open/closed)
- `labels`: Array of label names
- `assignees`: Array of assignee usernames
- `author`: Issue author username
- `created_at`, `updated_at`, `closed_at`: Timestamps
- `comments_count`: Number of comments
- `html_url`: GitHub issue URL
- `issue_number`: Issue number in the repository
- `milestone`: Associated milestone
- `reactions`: JSON object of reactions

### `github_issue_embeddings`

Stores semantic embeddings:
- `issue_id`: Reference to the issue
- `embedding_model`: Name of the embedding model used
- `embedding_vector`: 768-dimensional vector for similarity search
- `created_at`: When the embedding was computed

## Search Process

1. **Query Processing**: User query is converted to an embedding vector
2. **Vector Search**: pgvector finds the most similar issue embeddings
3. **Filtering**: Results are filtered by repository, state, author, labels
4. **Reranking**: Cross-encoder model improves result relevance
5. **Result Display**: Issues are shown with similarity scores and metadata

## Performance

- **Initial Fetch**: ~100 issues/second (respecting GitHub rate limits)
- **Embedding Computation**: ~10-50 issues/second (depending on model)
- **Search**: Sub-second response time for semantic search
- **Storage**: ~1-5 KB per issue + ~3 KB per embedding

## Rate Limiting

The fetcher automatically handles GitHub API rate limits:
- **Authenticated**: 5,000 requests/hour
- **Unauthenticated**: 60 requests/hour
- **Automatic waiting** when limits are reached
- **Respectful delays** between requests

## Examples

### Example 1: Find Performance Issues

```bash
uv run github_issues_searcher.py "performance slow laggy" --repository "microsoft/vscode" --state open
```

### Example 2: Find Documentation Requests

```bash
uv run github_issues_searcher.py "documentation missing unclear" --labels "documentation" "enhancement"
```

### Example 3: Find Security Issues

```bash
uv run github_issues_searcher.py "security vulnerability exploit" --state open --limit 10
```

### Example 4: Find Issues by Specific Author

```bash
uv run github_issues_searcher.py "feature request" --author "octocat" --type both
```

## Troubleshooting

### Common Issues

1. **No results found**: Ensure embeddings have been computed with `compute_github_embeddings.py`
2. **Rate limit errors**: The fetcher automatically handles this, but you can add a GitHub token for higher limits
3. **Database connection errors**: Check your database configuration in `config.json`
4. **Embedding model errors**: Ensure the embedding model can be downloaded (check internet connection)

### Debug Mode

For troubleshooting, you can add debug output:

```python
# In the search scripts, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- [ ] **Comment Search**: Include issue comments in semantic search
- [ ] **Pull Request Support**: Extend to pull requests
- [ ] **Advanced Filters**: Date ranges, milestone filtering
- [ ] **Bulk Operations**: Batch fetch from organization repositories
- [ ] **Web Interface**: Web-based search interface
- [ ] **API Endpoints**: REST API for integration with other tools

## Integration

The GitHub issues search can be integrated with:

- **CI/CD Pipelines**: Automatically search for related issues
- **Development Tools**: IDE plugins for issue discovery
- **Project Management**: Link issues to project tasks
- **Documentation**: Find related issues when writing docs

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the database schema and ensure tables exist
3. Verify your GitHub token has appropriate permissions
4. Check that pgvector extension is properly installed

---

**Happy Issue Hunting! 🐛🔍**
