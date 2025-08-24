#!/usr/bin/env python3
"""
Standalone Embedding Recomputation Script

This script recomputes embeddings for all emails in the database
without requiring access to the original Maildir folders.
"""

from database_manager import DatabaseManager
from config import Config
from sentence_transformers import SentenceTransformer


def main():
    """Main function to recompute embeddings."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recompute Email Embeddings')
    
    # Database options (optional if config file exists)
    parser.add_argument('--db-host', help='PostgreSQL database host (overrides config)')
    parser.add_argument('--db-port', type=int, help='PostgreSQL database port (overrides config)')
    parser.add_argument('--db-name', help='PostgreSQL database name (overrides config)')
    parser.add_argument('--db-user', help='PostgreSQL database user (overrides config)')
    parser.add_argument('--db-password', help='PostgreSQL database password (overrides config)')
    
    # Recomputation options
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing (default: 100)')
    parser.add_argument('--model', default='intfloat/e5-base', help='Embedding model to use (default: intfloat/e5-base)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually doing it')
    parser.add_argument('--show-stats', action='store_true', help='Show embedding statistics only')
    
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
        
        # Show current stats
        total_emails = db_manager.get_email_count()
        total_embeddings = db_manager.get_embedding_count()
        emails_without_embeddings = db_manager.get_emails_without_embeddings(limit=1000)
        
        print("Current Database Status:")
        print("=" * 40)
        print(f"Total emails: {total_emails}")
        print(f"Total embeddings: {total_embeddings}")
        print(f"Emails without embeddings: {len(emails_without_embeddings)}")
        if total_emails > 0:
            coverage = (total_embeddings / total_emails * 100)
            print(f"Coverage: {coverage:.1f}%")
        print()
        
        if args.show_stats:
            return 0
        
        if total_emails == 0:
            print("No emails found in database")
            return 0
        
        # Load embedding model
        print(f"Loading embedding model: {args.model}")
        try:
            embedding_model = SentenceTransformer(args.model)
            print(f"Model loaded successfully! ({embedding_model.get_sentence_embedding_dimension()} dimensions)")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return 1
        
        if args.dry_run:
            print("\nDRY RUN - No changes will be made")
            print(f"Would process {total_emails} emails in batches of {args.batch_size}")
            print(f"Would create {total_emails} new embeddings")
            return 0
        
        # Confirm before proceeding
        if total_embeddings > 0:
            print(f"\n⚠️  WARNING: This will delete {total_embeddings} existing embeddings!")
            response = input("Are you sure you want to continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Operation cancelled")
                return 0
        
        print(f"\nStarting embedding recomputation...")
        print(f"Model: {args.model}")
        print(f"Batch size: {args.batch_size}")
        print(f"Total emails to process: {total_emails}")
        print()
        
        # Perform recomputation
        stats = db_manager.recompute_embeddings(
            embedding_model=embedding_model,
            model_name=args.model.split('/')[-1],  # Extract model name from full path
            batch_size=args.batch_size,
            show_progress=True
        )
        
        print(f"\nRecomputation complete!")
        print("=" * 40)
        print(f"Processed: {stats['processed']}")
        print(f"Embeddings created: {stats['embeddings_created']}")
        print(f"Errors: {stats['errors']}")
        
        # Show updated stats
        updated_embeddings = db_manager.get_embedding_count()
        print(f"\nUpdated status:")
        print(f"Total embeddings: {updated_embeddings}")
        if total_emails > 0:
            updated_coverage = (updated_embeddings / total_emails * 100)
            print(f"Coverage: {updated_coverage:.1f}%")
        
        # Clean up
        db_manager.disconnect()
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
