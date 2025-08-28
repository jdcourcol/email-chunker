# Maildir Email Parser

A comprehensive Python library for reading, parsing, and analyzing emails from Maildir folders. Features multi-folder support, HTML to plain text conversion, and optional PostgreSQL database integration with sentence embeddings.

## 🚀 **Features**

- **Multi-folder Support**: Handle INBOX, Drafts, Sent, and other Maildir subdirectories
- **HTML Conversion**: Convert HTML emails to clean plain text with aggressive CSS cleaning
- **Comprehensive Parsing**: Extract headers, body, attachments, and metadata
- **Search & Analysis**: Search emails by subject, sender, or recipient
- **Database Integration**: Optional PostgreSQL storage with sentence embeddings
- **Semantic Search**: AI-powered email search using sentence transformers
- **pgvector Integration**: High-performance vector similarity search
- **Cross-Encoder Reranking**: Improved search relevance with MS-MARCO models
- **Hybrid Search**: Combine semantic and SQL search methods
- **Standalone Search**: Search database without Maildir access
- **Configurable Depth**: Adjustable semantic search coverage
- **Smart Prioritization**: Results ordered by search method relevance
- **PDF Document Search**: Index and search through PDF documents with semantic understanding
- **Multi-Format Support**: Handle emails and PDFs in a unified search interface
- **No External Dependencies**: Core functionality uses only Python standard library

## 📁 **Project Structure**

```
email-chunker/
├── main.py                 # Core MaildirParser class
├── enhanced_parser.py      # Database integration + embeddings
├── database_manager.py     # PostgreSQL operations
├── database_searcher.py    # Standalone search interface
├── search_emails.py        # Command-line search tool
├── reranker.py             # Cross-encoder reranking
├── recompute_embeddings.py # Embedding regeneration tool
├── config.py               # Configuration management
├── example_usage.py        # Usage examples
├── requirements.txt        # Dependencies
├── README.md              # This file
└── README_ENHANCED.md     # Enhanced features documentation
```

## 🚀 **Quick Start**

### **Basic Usage (No Database)**
```bash
# List all folders
python main.py /path/to/your/maildir --list-folders

# Process specific folder
python main.py /path/to/your/maildir --folder INBOX

# Search emails
python main.py /path/to/your/maildir --search "meeting"

# Search across all folders
python main.py /path/to/your/maildir --search "urgent" --search-all
```

### **Enhanced Usage (With Database)**
```bash
# Process all folders to database
python enhanced_parser.py /path/to/your/maildir \
    --db-host localhost \
    --db-name email_archive \
    --db-user your_username \
    --db-password your_password \
    --process-all

# Semantic search
python enhanced_parser.py /path/to/your/maildir \
    --db-host localhost \
    --db-name email_archive \
    --db-user your_username \
    --db-password your_password \
    --semantic-search "project deadline"
```

### **Advanced Search (Standalone)**
```bash
# Search directly from database (no Maildir required)
uv run search_emails.py --query "Azure migration" --type both --limit 10

# Semantic search with custom depth
uv run search_emails.py --query "project deadline" --type semantic --semantic-depth 200

# SQL search with case sensitivity
uv run search_emails.py --query "URGENT" --type sql --case-sensitive --show-sql

# PDF document search
uv run search_emails.py --query "contract terms" --type pdf --limit 5
```

### **PDF Document Indexing**
```bash
# Index PDF documents from a directory
uv run index_pdfs.py /path/to/documents --folder-name "contracts"

# Check what would be indexed (dry run)
uv run index_pdfs.py /path/to/documents --dry-run

# Index with custom database settings
uv run index_pdfs.py /path/to/documents \
    --db-host localhost \
    --db-name docs_db \
    --db-user username \
    --db-password password
```

## 📚 **Installation**

### **Basic Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/email-chunker.git
cd email-chunker

# Install dependencies (for enhanced features)
pip install -r requirements.txt
```

### **Using uv (Recommended)**
```bash
# Initialize with uv
uv init
uv add psycopg2-binary sentence-transformers numpy

# Or install from requirements
uv pip install -r requirements.txt
```

## 🔧 **Core Features**

### **MaildirParser Class**
- List and navigate Maildir folders
- Parse individual emails or entire folders
- Convert HTML to plain text
- Search emails by criteria
- Handle attachments and metadata

### **HTML Processing**
- Automatic HTML to plain text conversion
- Aggressive CSS cleaning (removes styles, scripts)
- Configurable cleaning levels
- Preserves email content while removing formatting

### **Multi-folder Support**
- Automatic folder detection
- Process specific folders or all folders
- Folder-specific email counts and statistics
- Cross-folder search capabilities

## 🗄️ **Database Integration (Enhanced)**

### **PostgreSQL Storage**
- Structured email storage with automatic table creation
- Efficient indexing for fast queries
- JSON storage for headers and attachments
- pgvector extension support for vector similarity search

### **Sentence Embeddings**
- **Model**: `e5-base` (intfloat/e5-base) - 768 dimensions
- **Quality**: High-performance semantic similarity search
- **Storage**: Optimized for pgvector and fallback to REAL[]
- **Coverage**: 100% email coverage with automatic embedding generation

### **Advanced Search Architecture**
- **Bi-Encoder**: Fast pgvector similarity search for initial retrieval
- **Cross-Encoder**: Reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Hybrid Search**: Combines semantic and SQL search methods
- **Smart Prioritization**: Results ordered by search method relevance

## 📖 **Usage Examples**

### **Basic Email Processing**
```python
from main import MaildirParser

# Initialize parser
parser = MaildirParser("/path/to/maildir")

# List available folders
folders = parser.list_folders()
print(f"Available folders: {folders}")

# Process specific folder
if parser.open_folder("INBOX"):
    emails = parser.get_all_emails()
    print(f"Found {len(emails)} emails in INBOX")
```

### **Enhanced Processing with Database**
```python
from enhanced_parser import EnhancedMaildirParser
from database_manager import DatabaseManager

# Setup database
db_manager = DatabaseManager({
    'host': 'localhost',
    'database': 'email_archive',
    'user': 'username',
    'password': 'password'
})

# Initialize enhanced parser
parser = EnhancedMaildirParser(
    '/path/to/maildir',
    db_manager=db_manager
)

# Process all folders
stats = parser.process_all_folders_to_database()
print(f"Processed {stats['saved']} emails")
```

### **Advanced Search Usage**
```python
from database_searcher import DatabaseSearcher
from reranker import create_reranker

# Initialize searcher with reranker
reranker = create_reranker()
searcher = DatabaseSearcher(db_manager, embedding_model, reranker)

# Hybrid search
results = searcher.search_emails_hybrid(
    query="Azure migration",
    search_type="both",
    limit=10
)

# Access results by type
semantic_results = results['semantic_results']
sql_results = results['sql_results']
combined_results = results['combined_results']

### **PDF Search Usage**
```python
from database_searcher import DatabaseSearcher
from reranker import create_reranker

# Initialize searcher with reranker
reranker = create_reranker()
searcher = DatabaseSearcher(db_manager, embedding_model, reranker)

# Search PDF documents
pdf_results = searcher.search_pdf_documents(
    query="contract terms",
    limit=10
)

# Access PDF metadata
for doc in pdf_results:
    print(f"File: {doc['file_name']}")
    print(f"Title: {doc['title']}")
    print(f"Author: {doc['author']}")
    print(f"Pages: {doc['page_count']}")
    print(f"Similarity: {doc['similarity_score']:.3f}")
```

## 🔍 **Advanced Search Capabilities**

### **Standalone Search (No Maildir Required)**
```bash
# Search directly from database
uv run search_emails.py --query "your search term" --type both --limit 10

# Semantic search only
uv run search_emails.py --query "your search term" --type semantic --limit 5

# SQL search only
uv run search_emails.py --query "your search term" --type sql --limit 5
```

### **Search Types**
- **`--type semantic`**: AI-powered semantic search using pgvector
- **`--type sql`**: Traditional text-based search using SQL LIKE/ILIKE
- **`--type both`**: Hybrid search combining both methods
- **`--type pdf`**: Search through indexed PDF documents using semantic similarity (no folder filtering)

### **Semantic Search Features**
- **pgvector Integration**: Database-level vector similarity search
- **Cross-Encoder Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for improved relevance
- **Configurable Depth**: `--semantic-depth` parameter (default: 100x limit)
- **Similarity Scores**: Bi-encoder similarity + cross-encoder relevance scores

### **Search Options**
```bash
--query "search term"           # Search query
--type semantic|sql|both|pdf   # Search type
--limit N                       # Maximum results
--folder FOLDER                 # Limit to specific folder (emails only)
--fields subject body sender    # SQL search fields
--case-sensitive                # Case-sensitive SQL search
--show-sql                      # Display SQL queries
--semantic-depth N              # Semantic search depth multiplier
```

### **Result Prioritization**
Results are automatically ordered by relevance:
1. **"Found via: both"** ⭐ (Highest - found by both semantic and SQL)
2. **"Found via: semantic"** (High - semantic meaning matches)
3. **"Found via: sql"** (Good - text pattern matches)

### **Traditional Search**
- Subject, sender, recipient matching
- Folder-based filtering
- Date range queries
- Exact and partial matching
- Case-sensitive/insensitive options

## 📊 **Performance**

### **Processing Speed**
- **Small Maildir** (<1000 emails): ~1-5 minutes
- **Medium Maildir** (1000-10000 emails): ~5-30 minutes
- **Large Maildir** (>10000 emails): ~30+ minutes

### **Search Performance**
- **pgvector Search**: Sub-second response for semantic queries
- **Cross-Encoder Reranking**: ~100-500ms per result (depending on model)
- **Hybrid Search**: Combines speed of pgvector with accuracy of reranking
- **Configurable Depth**: Balance between coverage and speed

### **Memory Usage**
- **Core parser**: ~50-100MB
- **With embeddings**: ~2.5GB recommended
- **Cross-encoder reranker**: ~200MB additional
- **Database**: Depends on email volume
- **pgvector**: Efficient vector storage and indexing

## 🛠️ **Configuration Options**

### **HTML Processing**
```bash
--no-html-convert      # Keep HTML as-is
--aggressive-clean     # Remove more CSS artifacts
```

### **Database Options**
```bash
--db-host HOST         # Database host
--db-name NAME         # Database name
--db-user USER         # Database user
--db-password PASS     # Database password
--no-embeddings        # Skip embedding generation
```

### **Processing Options**
```bash
--process-all          # Process all folders
--folder FOLDER        # Process specific folder
--semantic-search QUERY # Semantic search
--limit N              # Limit results
```

### **Search Options**
```bash
--query "search term"           # Search query
--type semantic|sql|both        # Search type
--limit N                       # Maximum results
--folder FOLDER                 # Limit to specific folder
--fields subject body sender    # SQL search fields
--case-sensitive                # Case-sensitive SQL search
--show-sql                      # Display SQL queries
--semantic-depth N              # Semantic search depth multiplier
```

## 🛠️ **Tools & Utilities**

### **Standalone Search Tool**
```bash
# Basic search
uv run search_emails.py --query "your query" --type both

# Advanced search with options
uv run search_emails.py \
    --query "Azure migration" \
    --type both \
    --limit 20 \
    --semantic-depth 200 \
    --show-sql \
    --case-sensitive

# PDF document search
uv run search_emails.py --query "contract terms" --type pdf --limit 10
```

### **Embedding Management**
```bash
# Recompute all embeddings
uv run recompute_embeddings.py --batch-size 100

# Check embedding status
uv run recompute_embeddings.py --show-stats

# Dry run (see what would be done)
uv run recompute_embeddings.py --dry-run
```

### **PDF Document Management**
```bash
# Index PDF documents
uv run index_pdfs.py /path/to/documents --folder-name "contracts"

# Check what would be indexed
uv run index_pdfs.py /path/to/documents --dry-run

# Index with progress updates
uv run index_pdfs.py /path/to/documents --verbose
```

### **Configuration Management**
```bash
# Setup configuration
python config.py setup

# View current config
python config.py show

# Test configuration
python config.py test
```

## 🚨 **Troubleshooting**

### **Common Issues**
1. **Maildir Path**: Ensure path points to root Maildir directory
2. **Database Connection**: Check PostgreSQL service and credentials
3. **Memory Issues**: Use `--no-embeddings` for large datasets
4. **HTML Conversion**: Use `--aggressive-clean` for problematic emails

### **Debug Mode**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 **Advanced Usage**

### **Custom Embedding Models**
```python
from sentence_transformers import SentenceTransformer

# Use different model
custom_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
parser = EnhancedMaildirParser(
    '/path/to/maildir',
    embedding_model=custom_model
)
```

### **Batch Processing**
```python
# Process folders individually
for folder in parser.list_folders():
    stats = parser.process_folder_to_database(folder)
    print(f"Folder {folder}: {stats['saved']} emails")
```

## 🔐 **Security & Privacy**

- **Local Processing**: All processing happens locally
- **Database Security**: Use secure database connections
- **Data Privacy**: Ensure compliance with email regulations
- **Access Control**: Limit database user permissions

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 **License**

MIT License - see LICENSE file for details.

## 📚 **Documentation**

- **This README**: Basic usage and features
- **[README_ENHANCED.md](README_ENHANCED.md)**: Database integration and embeddings
- **Example Code**: See `example_usage.py` for comprehensive examples

## 🙏 **Acknowledgments**

- **sentence-transformers**: For embedding models
- **PostgreSQL**: For robust database storage
- **Maildir**: For email storage format specification
- **Python Standard Library**: For core functionality

---

**For advanced features including database integration and semantic search, see [README_ENHANCED.md](README_ENHANCED.md)**
