#!/usr/bin/env python3
"""
PDF Document Indexer

Indexes PDF documents from a directory into the database for semantic search.
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_parser import PDFSearchIndexer
from database_manager import DatabaseManager
from config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Index PDF documents into the database for semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index PDFs from a documents folder
  uv run index_pdfs.py /path/to/documents --folder-name "contracts"
  
  # Index with custom database settings
  uv run index_pdfs.py /path/to/documents --db-host localhost --db-name docs_db
  
  # Check what would be indexed (dry run)
  uv run index_pdfs.py /path/to/documents --dry-run
  
  # Index with progress updates
  uv run index_pdfs.py /path/to/documents --verbose
        """
    )
    
    parser.add_argument(
        'directory',
        help='Directory containing PDF documents to index'
    )
    
    parser.add_argument(
        '--folder-name',
        default='documents',
        help='Folder name/category for the documents (default: documents)'
    )
    
    parser.add_argument(
        '--db-host',
        help='Database host (overrides config)'
    )
    
    parser.add_argument(
        '--db-name',
        help='Database name (overrides config)'
    )
    
    parser.add_argument(
        '--db-user',
        help='Database user (overrides config)'
    )
    
    parser.add_argument(
        '--db-password',
        help='Database password (overrides config)'
    )
    
    parser.add_argument(
        '--db-port',
        type=int,
        help='Database port (overrides config)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be indexed without actually indexing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress information'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"❌ Error: Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    db_config = config.get_db_config()
    if args.db_host:
        db_config['host'] = args.db_host
    if args.db_name:
        db_config['database'] = args.db_name
    if args.db_user:
        db_config['user'] = args.db_user
    if args.db_password:
        db_config['password'] = args.db_password
    if args.db_port:
        db_config['port'] = args.db_port
    
    # Initialize database manager
    db_manager = DatabaseManager(db_config)
    
    if not db_manager.connect():
        print("❌ Failed to connect to database")
        sys.exit(1)
    
    # Create tables if they don't exist
    if not db_manager.create_tables():
        print("❌ Failed to create database tables")
        sys.exit(1)
    
    # Initialize PDF indexer
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('intfloat/e5-base')
        pdf_indexer = PDFSearchIndexer(db_manager, embedding_model)
    except Exception as e:
        print(f"❌ Failed to initialize embedding model: {e}")
        sys.exit(1)
    
    print(f"📚 PDF Document Indexer")
    print(f"📁 Directory: {args.directory}")
    print(f"🏷️  Folder: {args.folder_name}")
    print(f"🔍 Database: {db_config['host']}:{db_config['port']}/{db_config['database']}")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No documents will be indexed")
        print()
        
        # Just scan and show what would be indexed
        from pdf_parser import PDFParser
        pdf_parser = PDFParser()
        pdf_files = pdf_parser.get_pdf_files_in_directory(args.directory)
        
        if not pdf_files:
            print("❌ No PDF files found in directory")
            sys.exit(0)
        
        print(f"📄 Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files[:10]:  # Show first 10
            file_name = os.path.basename(pdf_file)
            file_size = os.path.getsize(pdf_file)
            print(f"   📄 {file_name} ({file_size:,} bytes)")
        
        if len(pdf_files) > 10:
            print(f"   ... and {len(pdf_files) - 10} more files")
        
        print(f"\n💡 To actually index these documents, run without --dry-run")
        sys.exit(0)
    
    # Start indexing
    print("🚀 Starting PDF indexing...")
    print()
    
    try:
        # Index PDF documents
        stats = pdf_indexer.index_pdf_documents(
            directory_path=args.directory,
            folder_name=args.folder_name
        )
        
        print()
        print("📊 Indexing Summary:")
        print(f"   📁 Total PDFs found: {stats['total']}")
        print(f"   ✅ Successfully indexed: {stats['indexed']}")
        print(f"   ❌ Errors: {stats['errors']}")
        
        if stats['indexed'] > 0:
            print(f"\n🎉 Successfully indexed {stats['indexed']} PDF documents!")
            print(f"💡 You can now search these documents using:")
            print(f"   uv run search_emails.py --query 'your search term' --type semantic")
        else:
            print("\n⚠️  No documents were indexed successfully")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db_manager.disconnect()


if __name__ == "__main__":
    main()
