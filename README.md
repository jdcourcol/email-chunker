# Maildir Email Parser

A comprehensive Python library for reading, parsing, and analyzing emails from Maildir folders. Features multi-folder support, HTML to plain text conversion, and optional PostgreSQL database integration with sentence embeddings.

## 🚀 **Features**

- **Multi-folder Support**: Handle INBOX, Drafts, Sent, and other Maildir subdirectories
- **HTML Conversion**: Convert HTML emails to clean plain text with aggressive CSS cleaning
- **Comprehensive Parsing**: Extract headers, body, attachments, and metadata
- **Search & Analysis**: Search emails by subject, sender, or recipient
- **Database Integration**: Optional PostgreSQL storage with sentence embeddings
- **Semantic Search**: AI-powered email search using sentence transformers
- **No External Dependencies**: Core functionality uses only Python standard library

## 📁 **Project Structure**

```
email-chunker/
├── main.py                 # Core MaildirParser class
├── enhanced_parser.py      # Database integration + embeddings
├── database_manager.py     # PostgreSQL operations
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
- Structured email storage
- Automatic table creation
- Efficient indexing
- JSON storage for headers and attachments

### **Sentence Embeddings**
- Uses `e5-base` model (intfloat/e5-base)
- 768-dimensional embeddings
- High-quality semantic similarity search
- Excellent performance for semantic search

### **Semantic Search**
- Find emails by meaning, not just keywords
- Similarity scoring and ranking
- Context-aware search results
- Cross-folder semantic search

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

## 🔍 **Search Capabilities**

### **Traditional Search**
- Subject, sender, recipient matching
- Folder-based filtering
- Date range queries
- Exact and partial matching

### **Semantic Search**
- Concept-based search
- Meaning understanding
- Similarity scoring
- Cross-language support

## 📊 **Performance**

### **Processing Speed**
- **Small Maildir** (<1000 emails): ~1-5 minutes
- **Medium Maildir** (1000-10000 emails): ~5-30 minutes
- **Large Maildir** (>10000 emails): ~30+ minutes

### **Memory Usage**
- Core parser: ~50-100MB
- With embeddings: ~2.5GB recommended
- Database: Depends on email volume

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
