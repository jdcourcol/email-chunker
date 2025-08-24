#!/usr/bin/env python3
"""
Simple Email Search Script

This script demonstrates how to search emails stored in the database
without needing access to the original Maildir folders.
"""

from database_searcher import DatabaseSearcher, create_embedding_model
from database_manager import DatabaseManager
from config import Config


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
    parser.add_argument('--type', choices=['sql', 'semantic', 'both'], default='both',
                       help='Search type (default: both)')
    parser.add_argument('--fields', nargs='+', default=['subject', 'body', 'sender'],
                       help='Fields for SQL search (default: subject body sender)')
    parser.add_argument('--folder', help='Limit search to specific folder')
    parser.add_argument('--limit', type=int, default=10, help='Maximum results')
    parser.add_argument('--case-sensitive', action='store_true',
                       help='Perform case-sensitive SQL search (default: case-insensitive)')
    parser.add_argument('--show-sql', action='store_true',
                       help='Display the SQL query being executed')
    
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
        if args.type in ['semantic', 'both']:
            embedding_model = create_embedding_model()
            if not embedding_model:
                print("Warning: Could not load embedding model")
                if args.type == 'semantic':
                    args.type = 'sql'
                elif args.type == 'both':
                    args.type = 'sql'
        
        # Create searcher
        searcher = DatabaseSearcher(db_manager, embedding_model)
        
        # Perform search
        print(f"Searching for: '{args.query}'")
        print(f"Type: {args.type}")
        if args.folder:
            print(f"Folder: {args.folder}")
        print()
        
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
            print(f"   Folder: {email.get('folder', 'Unknown')}")
            print(f"   Date: {email.get('date_sent', 'Unknown')}")
            
            if 'similarity_score' in email:
                print(f"   Similarity: {email['similarity_score']:.3f}")
            
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
