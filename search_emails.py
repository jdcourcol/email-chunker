#!/usr/bin/env python3
"""
Simple Email Search Script

This script demonstrates how to search emails stored in the database
without needing access to the original Maildir folders.
"""

from database_searcher import DatabaseSearcher, create_embedding_model
from database_manager import DatabaseManager
from config import Config
from reranker import create_reranker
import urllib.parse


def main():
    """Simple search interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Search Emails in Database (No Maildir Required)')
    
    # Database options (optional if config file exists)
    parser.add_argument('--db-host', help='PostgreSQL database host (overrides config)')
    parser.add_argument('--db-port', type=int, help='PostgreSQL database port (overrides config)')
    parser.add_argument('--db-name', help='PostgreSQL database name (overrides config)')
    parser.add_argument('--db-user', help='PostgreSQL database user (overrides config)')
    parser.add_argument('--db-password', help='PostgreSQL database password (overrides config)')
    
    # Search options
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--type', choices=['sql', 'semantic', 'both', 'pdf'], default='both',
                       help='Search type (default: both)')
    parser.add_argument('--fields', nargs='+', default=['subject', 'body', 'sender'],
                       help='Fields for SQL search (default: subject body sender)')
    parser.add_argument('--folder', help='Limit search to specific folder')
    parser.add_argument('--limit', type=int, default=10, help='Maximum results')
    parser.add_argument('--case-sensitive', action='store_true',
                       help='Perform case-sensitive SQL search (default: case-insensitive)')
    parser.add_argument('--show-sql', action='store_true',
                       help='Display the SQL query being executed')
    parser.add_argument('--semantic-depth', type=int, default=100,
                       help='Semantic search depth multiplier (default: 100, max: 1000)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Setup database connection (command line args override config)
    db_params = config.get_db_config()
    
    # Override with command line arguments if provided
    if args.db_host:
        db_params['host'] = args.db_host
    if args.db_port:
        db_params['port'] = args.db_port
    if args.db_name:
        db_params['database'] = args.db_name
    if args.db_user:
        db_params['user'] = args.db_user
    if args.db_password:
        db_params['password'] = args.db_password
    
    # Check if we have complete database configuration
    if not all(db_params.get(key) for key in ['host', 'database', 'user', 'password']):
        print("Error: Incomplete database configuration")
        print("Please run: python config.py setup")
        print("Or provide all database parameters via command line")
        return 1
    
    try:
        # Connect to database
        db_manager = DatabaseManager(db_params)
        if not db_manager.connect():
            print("Failed to connect to database")
            return 1
        
        # Load embedding model if needed
        embedding_model = None
        if args.type in ['semantic', 'both', 'pdf']:
            embedding_model = create_embedding_model()
            if not embedding_model:
                print("Warning: Could not load embedding model")
                if args.type == 'semantic':
                    args.type = 'sql'
                elif args.type == 'both':
                    args.type = 'sql'
                elif args.type == 'pdf':
                    print("Error: PDF search requires embedding model")
                    return 1
        
        # Load reranker for better result relevance
        reranker = create_reranker()
        
        # Create searcher
        searcher = DatabaseSearcher(db_manager, embedding_model, reranker)
        
        # Set semantic search depth
        searcher.semantic_depth_multiplier = min(args.semantic_depth, 1000)  # Cap at 1000
        
        # Perform search
        print(f"Searching for: '{args.query}'")
        print(f"Type: {args.type}")
        if args.folder:
            print(f"Folder: {args.folder}")
        print()
        
        if args.type == 'pdf':
            # PDF document search
            results = searcher.search_pdf_documents(
                query=args.query,
                limit=args.limit
            )
            
            if not results:
                print("No PDF documents found")
                return 0
            
            print(f"Found {len(results)} PDF documents:")
            print()
            
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.get('file_name', 'Unknown file')}")
                print(f"   Title: {doc.get('title', 'No title')}")
                print(f"   Author: {doc.get('author', 'Unknown')}")
                print(f"   Folder: {doc.get('folder_name', 'Unknown')}")
                print(f"   Pages: {doc.get('page_count', 'Unknown')}")
                print(f"   Size: {doc.get('file_size', 0):,} bytes")
                print(f"   Similarity: {doc.get('similarity_score', 0):.3f}")
                
                if 'cross_encoder_score' in doc:
                    print(f"   Cross-Encoder: {doc['cross_encoder_score']:.3f}")
                
                # Show snippet of content
                content = doc.get('content', '')
                if content:
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    print(f"   Content: {snippet}")
                print()
            
            return 0
        else:
            # Email search
            results = searcher.search_emails_hybrid(
                query=args.query,
                search_type=args.type,
                search_fields=args.fields,
                limit=args.limit,
                folder=args.folder,
                case_sensitive=args.case_sensitive,
                show_sql=args.show_sql
            )
        
        if not results:
            print("No results found")
            return 0
        
        # Display results
        metadata = results['search_metadata']
        print(f"Found {metadata['total_results']} results:")
        print(f"  SQL: {metadata.get('sql_count', 0)}")
        print(f"  Semantic: {metadata.get('semantic_count', 0)}")
        print()
        
        # Show results
        if args.type == 'both' and results['combined_results']:
            emails = results['combined_results']
        elif args.type == 'sql' and results['sql_results']:
            emails = results['sql_results']
        elif args.type == 'semantic' and results['semantic_results']:
            emails = results['semantic_results']
        else:
            emails = []
        
        for i, email in enumerate(emails, 1):
            print(f"{i}. {email.get('subject', 'No subject')}")
            print(f"   From: {email.get('sender', 'Unknown')}")
            print(f"   To: {email.get('recipient', 'Unknown')}")
            print(f"   Message ID: {email.get('message_id', 'Unknown')}")
            print(f"   Database ID: {email.get('id', 'Unknown')}")
            print(f"   Folder: {email.get('folder', 'Unknown')}")
            print(f"   Date: {email.get('date_sent', 'Unknown')}")
            
            # Add URL for message retrieval
            message_id = email.get('message_id', '')
            if message_id and message_id != 'Unknown':
                # Remove angle brackets from message ID for cleaner URLs
                clean_message_id = message_id.replace('<', '').replace('>', '')
                
                # Only encode if the message ID contains characters that need encoding
                if any(char in clean_message_id for char in [' ', '\n', '\t', '¶']):
                    encoded_message_id = urllib.parse.quote(clean_message_id, safe='')
                else:
                    encoded_message_id = clean_message_id
                url = f"http://localhost:8000/get_message/?messageid={encoded_message_id}"
                print(f"   URL: {url}")
            
            if 'similarity_score' in email:
                print(f"   Similarity: {email['similarity_score']:.3f}")
            
            if 'cross_encoder_score' in email:
                print(f"   Cross-Encoder: {email['cross_encoder_score']:.3f}")
            
            search_type = email.get('search_type', 'unknown')
            print(f"   Found via: {search_type}")
            print()
        
        # Clean up
        db_manager.disconnect()
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
