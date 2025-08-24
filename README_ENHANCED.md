# Enhanced Maildir Email Parser with Database Integration

A powerful Python library for reading, parsing, and storing emails from Maildir folders with PostgreSQL database integration and sentence embeddings for semantic search.

## 🚀 **New Features**

- **PostgreSQL Database Storage**: Store all parsed emails in a structured database
- **Sentence Embeddings**: Generate embeddings using `all-MiniLM-L6-v2` model
- **Semantic Search**: Find emails by meaning, not just keywords
- **Multi-folder Support**: Handle INBOX, Drafts, Sent, and other folders
- **HTML to Plain Text**: Clean conversion with aggressive CSS cleaning
- **Batch Processing**: Process entire Maildir structures efficiently

## 📋 **Requirements**

### Python Dependencies
```bash
pip install -r requirements.txt
```

### System Requirements
- PostgreSQL 12+ database server
- Python 3.8+
- Sufficient RAM for embedding model (~2GB recommended)

## 🗄️ **Database Schema**

The system creates two main tables:

### `emails` Table
```sql
CREATE TABLE emails (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(500) UNIQUE,
    subject TEXT,
    sender VARCHAR(500),
    recipient VARCHAR(500),
    date_sent TIMESTAMP,
    folder VARCHAR(100),
    content_type VARCHAR(100),
    body TEXT,
    headers JSONB,
    attachments JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `email_embeddings` Table
```sql
CREATE TABLE email_embeddings (
    id SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
    embedding_vector REAL[],
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🚀 **Quick Start**

### 1. **Setup Database**
```bash
# Create PostgreSQL database
createdb email_archive

# Or use existing database
```

### 2. **Process All Folders**
```bash
python enhanced_parser.py /path/to/your/maildir \
    --db-host localhost \
    --db-name email_archive \
    --db-user your_username \
    --db-password your_password \
    --process-all
```

### 3. **Process Specific Folder**
```bash
python enhanced_parser.py /path/to/your/maildir \
    --db-host localhost \
    --db-name email_archive \
    --db-user your_username \
    --db-password your_password \
    --folder INBOX
```

### 4. **Semantic Search**
```bash
python enhanced_parser.py /path/to/your/maildir \
    --db-host localhost \
    --db-name email_archive \
    --db-user your_username \
    --db-password your_password \
    --semantic-search "meeting schedule"
```

## 📚 **Usage Examples**

### **Basic Database Processing**
```python
from enhanced_parser import EnhancedMaildirParser, DatabaseManager
from database_manager import create_embedding_model

# Setup database connection
db_params = {
    'host': 'localhost',
    'port': 5432,
    'database': 'email_archive',
    'user': 'your_username',
    'password': 'your_password'
}

db_manager = DatabaseManager(db_params)
db_manager.create_tables()

# Load embedding model
embedding_model = create_embedding_model()

# Initialize parser
parser = EnhancedMaildirParser(
    '/path/to/maildir',
    convert_html=True,
    aggressive_clean=True,
    db_manager=db_manager,
    embedding_model=embedding_model
)

# Process all folders
stats = parser.process_all_folders_to_database()
print(f"Processed {stats['saved']} emails with {stats['embeddings']} embeddings")
```

### **Semantic Search**
```python
# Search for semantically similar emails
results = parser.search_emails_semantic("project deadline", limit=10)

for email in results:
    similarity = email.get('similarity_score', 0)
    print(f"Subject: {email['subject']}")
    print(f"Similarity: {similarity:.3f}")
    print(f"From: {email['sender']}")
    print("-" * 50)
```

### **Database Statistics**
```python
# Get database statistics
stats = parser.get_database_stats()
print(f"Total emails: {stats['total_emails']}")
print("Emails by folder:")
for folder, count in stats['folders'].items():
    print(f"  {folder}: {count}")
```

## 🔧 **Command Line Options**

### **Required Arguments**
- `maildir_path`: Path to Maildir root folder

### **Database Options**
- `--db-host`: PostgreSQL host
- `--db-name`: Database name
- `--db-user`: Database username
- `--db-password`: Database password
- `--db-port`: Database port (default: 5432)

### **Processing Options**
- `--process-all`: Process all folders to database
- `--folder FOLDER`: Process specific folder only
- `--no-embeddings`: Skip computing embeddings
- `--semantic-search QUERY`: Perform semantic search

## 🧠 **Embedding Model**

### **Model Details**
- **Model**: `e5-base` (intfloat/e5-base)
- **Dimensions**: 768
- **Performance**: High quality embeddings, excellent for semantic search
- **Memory**: ~200MB

### **Text Processing for Embeddings**
```python
# Creates embedding from subject + body
text_for_embedding = f"{subject} {body}"
embedding = model.encode(text_for_embedding).tolist()
```

### **Similarity Computation**
```python
# Cosine similarity between query and email embeddings
similarity = np.dot(query_embedding, email_embedding) / (
    np.linalg.norm(query_embedding) * np.linalg.norm(email_embedding)
)
```

## 📊 **Performance Considerations**

### **Database Performance**
- **Indexes**: Automatic creation on message_id, folder, date_sent
- **Batch Processing**: Efficient bulk inserts
- **Connection Pooling**: Single connection per session

### **Embedding Performance**
- **Model Loading**: ~3-8 seconds first time
- **Inference**: ~15-80ms per email (depending on text length)
- **Memory**: ~2.5GB RAM recommended for large datasets

### **Processing Speed**
- **Small Maildir** (<1000 emails): ~1-5 minutes
- **Medium Maildir** (1000-10000 emails): ~5-30 minutes
- **Large Maildir** (>10000 emails): ~30+ minutes

## 🔍 **Search Capabilities**

### **Traditional Search**
- Subject, sender, recipient matching
- Date range filtering
- Folder-based filtering

### **Semantic Search**
- **Meaning-based**: Find emails about concepts, not just keywords
- **Similarity Scoring**: Ranked results by relevance
- **Context Understanding**: Understands email content semantics

### **Example Queries**
```bash
# Find emails about meetings
--semantic-search "meeting schedule"

# Find emails about deadlines
--semantic-search "project deadline urgent"

# Find emails about technical issues
--semantic-search "system error troubleshooting"
```

## 🛠️ **Advanced Usage**

### **Custom Embedding Models**
```python
from sentence_transformers import SentenceTransformer

# Use different model
custom_model = SentenceTransformer('intfloat/e5-large-v2')  # Larger, higher quality
# Or use multilingual model
# custom_model = SentenceTransformer('intfloat/multilingual-e5-base')
parser = EnhancedMaildirParser(
    '/path/to/maildir',
    embedding_model=custom_model,
    # ... other params
)
```

### **Batch Processing with Progress**
```python
# Process folders with progress tracking
folders = parser.list_folders()
total_processed = 0

for folder in folders:
    print(f"Processing {folder}...")
    stats = parser.process_folder_to_database(folder)
    total_processed += stats['saved']
    print(f"Progress: {total_processed} emails processed")
```

### **Database Maintenance**
```python
# Check database health
db_stats = parser.get_database_stats()

# Rebuild embeddings if needed
if db_stats.get('total_emails', 0) > 0:
    # Process existing emails to recreate embeddings
    parser.process_all_folders_to_database()
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Database Connection Failed**
   - Check PostgreSQL service is running
   - Verify connection parameters
   - Ensure database exists and user has permissions

2. **Model Loading Failed**
   - Install sentence-transformers: `pip install sentence-transformers`
   - Check available memory (>2GB recommended)
   - Verify internet connection for model download

3. **Memory Issues**
   - Process folders individually instead of all at once
   - Use `--no-embeddings` for initial processing
   - Increase system RAM or use swap

4. **Performance Issues**
   - Check database indexes are created
   - Monitor system resources during processing
   - Consider processing during off-peak hours

### **Debug Mode**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose output
parser = EnhancedMaildirParser(
    '/path/to/maildir',
    db_manager=db_manager,
    embedding_model=embedding_model
)
```

## 📈 **Scaling Considerations**

### **Large Datasets**
- **Incremental Processing**: Process new folders only
- **Batch Sizes**: Process folders individually
- **Database Optimization**: Regular VACUUM and ANALYZE

### **Production Deployment**
- **Connection Pooling**: Use connection pool for multiple users
- **Background Processing**: Use Celery or similar for async processing
- **Monitoring**: Track processing times and database performance

## 🔐 **Security Considerations**

- **Database Credentials**: Store securely, use environment variables
- **Network Security**: Use SSL connections to database
- **Access Control**: Limit database user permissions
- **Data Privacy**: Ensure email data handling complies with regulations

## 📝 **API Reference**

### **EnhancedMaildirParser Class**

#### **Methods**
- `save_emails_to_database(emails, compute_embeddings=True)`
- `process_folder_to_database(folder_name, compute_embeddings=True)`
- `process_all_folders_to_database(compute_embeddings=True)`
- `search_emails_semantic(query, limit=10)`
- `get_database_stats()`

### **DatabaseManager Class**

#### **Methods**
- `connect()`, `disconnect()`
- `create_tables()`
- `save_email(email_data)`
- `save_embedding(email_id, embedding, model_name)`
- `search_emails_semantic(query, limit)`

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 **License**

MIT License - see LICENSE file for details.

## 🙏 **Acknowledgments**

- **sentence-transformers**: For the embedding models
- **PostgreSQL**: For robust database storage
- **Maildir**: For the email storage format specification
