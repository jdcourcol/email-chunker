#!/usr/bin/env python3
"""
Enhanced Maildir Email Parser with Database Integration

This script extends the basic MaildirParser with PostgreSQL database integration
and sentence embeddings using sentence-transformers.
"""

from main import MaildirParser, html_to_plain_text
from database_manager import DatabaseManager
from config import Config
import numpy as np
from typing import List, Dict, Any, Optional


class EnhancedMaildirParser(MaildirParser):
    """Enhanced MaildirParser with database integration and embeddings."""
    
    def __init__(self, maildir_path: Optional[str] = None, convert_html: bool = True, aggressive_clean: bool = False, 
                 db_manager: Optional[DatabaseManager] = None, embedding_model=None):
        """
        Initialize the EnhancedMaildirParser.
        
        Args:
            maildir_path: Optional path to the Maildir folder (only needed for processing emails)
            convert_html: Whether to convert HTML content to plain text (default: True)
            aggressive_clean: Whether to use aggressive CSS/HTML cleaning (default: False)
            db_manager: Optional database manager for saving emails
            embedding_model: Optional sentence transformer model for embeddings
        """
        # Only initialize MaildirParser if maildir_path is provided
        if maildir_path:
            super().__init__(maildir_path, convert_html, aggressive_clean)
        else:
            # For search-only operations, we don't need to initialize the parent class
            self.maildir_path = None
            self.convert_html = convert_html
            self.aggressive_clean = aggressive_clean
        
        self.db_manager = db_manager
        self.embedding_model = embedding_model
    
    def is_search_only(self) -> bool:
        """Check if this parser is in search-only mode (no Maildir access)."""
        return self.maildir_path is None
    
    def save_emails_to_database(self, emails: List[Dict[str, Any]], 
                               compute_embeddings: bool = True) -> Dict[str, int]:
        """
        Save emails to database and optionally compute embeddings.
        
        Args:
            emails: List of parsed email dictionaries
            compute_embeddings: Whether to compute and save embeddings
            
        Returns:
            Dictionary with save statistics
        """
        if not self.db_manager:
            print("No database manager configured")
            return {'saved': 0, 'skipped': 0, 'embeddings': 0}
        
        saved_count = 0
        skipped_count = 0
        embedding_count = 0
        
        for email_data in emails:
            message_id = email_data.get('message_id', '')
            if not message_id:
                print("Skipping email without message ID")
                skipped_count += 1
                continue
            
            # Check if email already exists
            if self.db_manager.email_exists(message_id):
                print(f"Email already exists: {message_id[:50]}...")
                skipped_count += 1
                continue
            
            # Save email to database
            email_id = self.db_manager.save_email(email_data)
            if email_id:
                saved_count += 1
                print(f"Saved email: {email_data.get('subject', 'No subject')[:50]}...")
                
                # Compute and save embedding if requested
                if compute_embeddings and self.embedding_model:
                    try:
                        # Create text for embedding (subject + body)
                        text_for_embedding = f"{email_data.get('subject', '')} {email_data.get('body', '')}"
                        text_for_embedding = text_for_embedding.strip()
                        
                        if text_for_embedding:
                            # Compute embedding
                            embedding = self.embedding_model.encode(text_for_embedding).tolist()
                            
                            # Save embedding to database
                            if self.db_manager.save_embedding(email_id, embedding, 'e5-base'):
                                embedding_count += 1
                                print(f"  Saved embedding for email ID {email_id}")
                            else:
                                print(f"  Failed to save embedding for email ID {email_id}")
                    except Exception as e:
                        print(f"  Error computing embedding for email ID {email_id}: {e}")
            else:
                print(f"Failed to save email: {email_data.get('subject', 'No subject')[:50]}...")
                skipped_count += 1
        
        return {
            'saved': saved_count,
            'skipped': skipped_count,
            'embeddings': embedding_count
        }
    
    def process_folder_to_database(self, folder_name: str, compute_embeddings: bool = True) -> Dict[str, int]:
        """
        Process all emails from a specific folder and save to database.
        
        Args:
            folder_name: Name of the folder to process
            compute_embeddings: Whether to compute and save embeddings
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"Processing folder: {folder_name}")
        
        if not self.open_folder(folder_name):
            return {'saved': 0, 'skipped': 0, 'embeddings': 0}
        
        emails = self.get_all_emails()
        print(f"Found {len(emails)} emails in {folder_name}")
        
        return self.save_emails_to_database(emails, compute_embeddings)
    
    def process_all_folders_to_database(self, compute_embeddings: bool = True) -> Dict[str, int]:
        """
        Process all folders and save emails to database.
        
        Args:
            compute_embeddings: Whether to compute and save embeddings
            
        Returns:
            Dictionary with processing statistics
        """
        total_stats = {'saved': 0, 'skipped': 0, 'embeddings': 0}
        
        for folder_name in self.list_folders():
            folder_stats = self.process_folder_to_database(folder_name, compute_embeddings)
            
            total_stats['saved'] += folder_stats['saved']
            total_stats['skipped'] += folder_stats['skipped']
            total_stats['embeddings'] += folder_stats['embeddings']
            
            print(f"Folder {folder_name}: {folder_stats['saved']} saved, {folder_stats['skipped']} skipped")
        
        return total_stats
    
    def search_emails_semantic(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search emails using semantic similarity with embeddings.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            List of emails with similarity scores
        """
        if not self.db_manager or not self.embedding_model:
            print("Database manager or embedding model not configured")
            return []
        
        try:
            # Compute query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Get emails with embeddings from database
            emails_with_embeddings = self.db_manager.search_emails_semantic(query, limit * 2)
            
            if not emails_with_embeddings:
                return []
            
            # Compute similarity scores
            results = []
            for email in emails_with_embeddings:
                if 'embedding_vector' in email and email['embedding_vector']:
                    # Convert embedding to numpy array for similarity computation
                    email_embedding = np.array(email['embedding_vector'])
                    query_embedding_array = np.array(query_embedding)
                    
                    # Compute cosine similarity
                    similarity = np.dot(email_embedding, query_embedding_array) / (
                        np.linalg.norm(email_embedding) * np.linalg.norm(query_embedding_array)
                    )
                    
                    # Add similarity score to email data
                    email_copy = dict(email)
                    email_copy['similarity_score'] = float(similarity)
                    results.append(email_copy)
            
            # Sort by similarity score (descending) and return top results
            results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def search_emails_hybrid(self, query: str, search_type: str = 'both', 
                           search_fields: List[str] = None, limit: int = 100,
                           folder: str = None, case_sensitive: bool = False, show_sql: bool = False) -> Dict[str, Any]:
        """
        Hybrid search combining SQL LIKE and semantic search.
        
        Args:
            query: Search query text
            search_type: 'sql', 'semantic', or 'both'
            search_fields: Fields for SQL search (default: ['subject', 'body', 'sender'])
            limit: Maximum results per search type
            folder: Optional folder to limit search to
            case_sensitive: Whether to perform case-sensitive SQL search (default: False)
            show_sql: Whether to display the SQL query being executed
            
        Returns:
            Dictionary with search results and metadata
        """
        if not self.db_manager:
            print("Database manager not configured")
            return {}
        
        try:
            # Use database manager's hybrid search
            results = self.db_manager.search_emails_hybrid(
                query, search_type, search_fields, limit, folder, 0.1, case_sensitive, show_sql
            )
            
            # If semantic search is requested but no embeddings available, fall back to SQL
            if search_type in ['semantic', 'both'] and results['search_metadata'].get('semantic_count', 0) == 0:
                print("No embeddings available, falling back to SQL search")
                if search_type == 'semantic':
                    search_type = 'sql'
                results = self.db_manager.search_emails_hybrid(
                    query, search_type, search_fields, limit, folder, 0.1, False, False
                )
            
            return results
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return {}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.db_manager:
            return {}
        
        try:
            total_emails = self.db_manager.get_email_count()
            
            # Get folder statistics
            folders = self.list_folders()
            folder_stats = {}
            
            for folder in folders:
                emails_in_folder = self.db_manager.get_emails_by_folder(folder, limit=1000)
                folder_stats[folder] = len(emails_in_folder)
            
            return {
                'total_emails': total_emails,
                'folders': folder_stats,
                'database_connected': True
            }
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {'database_connected': False, 'error': str(e)}
    
    def recompute_all_embeddings(self, batch_size: int = 100, show_progress: bool = True) -> Dict[str, int]:
        """
        Recompute embeddings for all emails in the database.
        
        Args:
            batch_size: Number of emails to process in each batch
            show_progress: Whether to show progress updates
            
        Returns:
            Dictionary with recomputation statistics
        """
        if not self.db_manager or not self.embedding_model:
            print("Database manager or embedding model not configured")
            return {'processed': 0, 'embeddings_created': 0, 'errors': 0}
        
        try:
            print("Starting embedding recomputation...")
            print(f"Model: {self.embedding_model.get_sentence_embedding_dimension()} dimensions")
            
            stats = self.db_manager.recompute_embeddings(
                self.embedding_model,
                model_name='e5-base',
                batch_size=batch_size,
                show_progress=show_progress
            )
            
            return stats
            
        except Exception as e:
            print(f"Error during embedding recomputation: {e}")
            return {'processed': 0, 'embeddings_created': 0, 'errors': 0}
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about embeddings in the database.
        
        Returns:
            Dictionary with embedding statistics
        """
        if not self.db_manager:
            return {}
        
        try:
            total_emails = self.db_manager.get_email_count()
            total_embeddings = self.db_manager.get_embedding_count()
            emails_without_embeddings = self.db_manager.get_emails_without_embeddings(limit=1000)
            
            return {
                'total_emails': total_emails,
                'total_embeddings': total_embeddings,
                'emails_without_embeddings': len(emails_without_embeddings),
                'coverage_percentage': (total_embeddings / total_emails * 100) if total_emails > 0 else 0,
                'embedding_model': 'e5-base'
            }
            
        except Exception as e:
            print(f"Error getting embedding stats: {e}")
            return {}


def create_embedding_model():
    """
    Create and return a sentence transformer model.
    
    Returns:
        SentenceTransformer model or None if failed
    """
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence transformer model 'e5-base'...")
        model = SentenceTransformer('intfloat/e5-base')
        print("Model loaded successfully!")
        return model
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def main():
    """Example usage of the enhanced parser."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Maildir Parser with Database Integration')
    parser.add_argument('maildir_path', nargs='?', help='Path to the Maildir root folder (optional for search)')
    
    # Database options (optional if config file exists)
    parser.add_argument('--db-host', help='PostgreSQL database host (overrides config)')
    parser.add_argument('--db-port', type=int, help='PostgreSQL database port (overrides config)')
    parser.add_argument('--db-name', help='PostgreSQL database name (overrides config)')
    parser.add_argument('--db-user', help='PostgreSQL database user (overrides config)')
    parser.add_argument('--db-password', help='PostgreSQL database password (overrides config)')
    
    # Processing options
    parser.add_argument('--process-all', action='store_true', help='Process all folders to database')
    parser.add_argument('--folder', help='Process specific folder only')
    parser.add_argument('--no-embeddings', action='store_true', help='Skip computing embeddings')
    parser.add_argument('--recompute-embeddings', action='store_true', help='Recompute all embeddings in database')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for embedding recomputation')
    
    # Search options
    parser.add_argument('--search', help='Search query')
    parser.add_argument('--search-type', choices=['sql', 'semantic', 'both'], default='both',
                       help='Search type: sql (LIKE), semantic (embeddings), or both (default: both)')
    parser.add_argument('--search-fields', nargs='+', 
                       default=['subject', 'body', 'sender'],
                       help='Fields to search in for SQL search (default: subject body sender)')
    parser.add_argument('--case-sensitive', action='store_true',
                       help='Perform case-sensitive SQL search (default: case-insensitive)')
    parser.add_argument('--show-sql', action='store_true',
                       help='Display the SQL query being executed')
    parser.add_argument('--semantic-search', help='Perform semantic search with query (legacy option)')
    
    args = parser.parse_args()
    
    try:
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
        
        # Initialize database manager
        db_manager = DatabaseManager(db_params)
        if not db_manager.create_tables():
            print("Failed to create database tables")
            return 1
        
        print("Database connection established and tables created")
        
        # Initialize embedding model
        embedding_model = None
        if not args.no_embeddings:
            embedding_model = create_embedding_model()
            if not embedding_model:
                print("Warning: Continuing without embeddings")
        
        # Initialize enhanced parser
        # For search operations, we don't need the Maildir path
        maildir_path = args.maildir_path if hasattr(args, 'maildir_path') and args.maildir_path else None
        
        # Check if Maildir path is required for the requested operation
        if not maildir_path and any([args.process_all, args.folder]):
            print("Error: Maildir path is required for processing operations")
            print("Use: python enhanced_parser.py /path/to/maildir [options]")
            return 1
        
        enhanced_parser = EnhancedMaildirParser(
            maildir_path,
            convert_html=True,
            aggressive_clean=True,
            db_manager=db_manager,
            embedding_model=embedding_model
        )
        
        # Process all folders if requested
        if args.process_all:
            print("Processing all folders to database...")
            stats = enhanced_parser.process_all_folders_to_database(compute_embeddings=not args.no_embeddings)
            
            print(f"\nProcessing complete:")
            print(f"  Total emails saved: {stats['saved']}")
            print(f"  Total emails skipped: {stats['skipped']}")
            print(f"  Total embeddings created: {stats['embeddings']}")
            
            # Show database stats
            db_stats = enhanced_parser.get_database_stats()
            print(f"  Total emails in database: {db_stats.get('total_emails', 0)}")
            
            return 0
        
        # Process specific folder if requested
        if args.folder:
            print(f"Processing folder: {args.folder}")
            stats = enhanced_parser.process_folder_to_database(args.folder, compute_embeddings=not args.no_embeddings)
            
            print(f"Folder processing complete:")
            print(f"  Emails saved: {stats['saved']}")
            print(f"  Emails skipped: {stats['skipped']}")
            print(f"  Embeddings created: {stats['embeddings']}")
            
            return 0
        
        # Recompute embeddings if requested
        if args.recompute_embeddings:
            if not embedding_model:
                print("Error: Embedding model required for recomputation")
                return 1
            
            print("Starting embedding recomputation...")
            print(f"Batch size: {args.batch_size}")
            
            # Show current embedding stats
            embedding_stats = enhanced_parser.get_embedding_stats()
            print(f"\nCurrent embedding status:")
            print(f"  Total emails: {embedding_stats.get('total_emails', 0)}")
            print(f"  Total embeddings: {embedding_stats.get('total_embeddings', 0)}")
            print(f"  Coverage: {embedding_stats.get('coverage_percentage', 0):.1f}%")
            print()
            
            # Perform recomputation
            stats = enhanced_parser.recompute_all_embeddings(
                batch_size=args.batch_size,
                show_progress=True
            )
            
            print(f"\nRecomputation complete:")
            print(f"  Processed: {stats['processed']}")
            print(f"  Embeddings created: {stats['embeddings_created']}")
            print(f"  Errors: {stats['errors']}")
            
            # Show updated stats
            updated_stats = enhanced_parser.get_embedding_stats()
            print(f"\nUpdated embedding status:")
            print(f"  Total emails: {updated_stats.get('total_emails', 0)}")
            print(f"  Total embeddings: {updated_stats.get('total_embeddings', 0)}")
            print(f"  Coverage: {updated_stats.get('coverage_percentage', 0):.1f}%")
            
            return 0
        
        # Search if requested
        if args.search:
            print(f"Performing {args.search_type} search for: '{args.search}'")
            
            # Use hybrid search
            results = enhanced_parser.search_emails_hybrid(
                query=args.search,
                search_type=args.search_type,
                search_fields=args.search_fields,
                limit=20,
                folder=args.folder,
                case_sensitive=args.case_sensitive,
                show_sql=args.show_sql
            )
            
            if not results:
                print("No search results found")
                return 0
            
            # Display search metadata
            metadata = results['search_metadata']
            print(f"\nSearch Results:")
            print(f"  Search Type: {metadata['search_type']}")
            print(f"  Folder: {metadata.get('folder', 'All folders')}")
            print(f"  SQL Results: {metadata.get('sql_count', 0)}")
            print(f"  Semantic Results: {metadata.get('semantic_count', 0)}")
            print(f"  Total Results: {metadata['total_results']}")
            
            # Display results
            if args.search_type == 'both' and results['combined_results']:
                emails_to_show = results['combined_results']
            elif args.search_type == 'sql' and results['sql_results']:
                emails_to_show = results['sql_results']
            elif args.search_type == 'semantic' and results['semantic_results']:
                emails_to_show = results['semantic_results']
            else:
                emails_to_show = []
            
            if emails_to_show:
                print(f"\nTop {len(emails_to_show)} results:")
                for i, email in enumerate(emails_to_show, 1):
                    search_type = email.get('search_type', 'unknown')
                    similarity = email.get('similarity_score', 0)
                    
                    print(f"{i}. {email.get('subject', 'No subject')}")
                    print(f"   From: {email.get('sender', 'Unknown')}")
                    print(f"   Folder: {email.get('folder', 'Unknown')}")
                    print(f"   Date: {email.get('date_sent', 'Unknown')}")
                    if search_type == 'semantic' and similarity > 0:
                        print(f"   Similarity: {similarity:.3f}")
                    print(f"   Search Type: {search_type}")
                    print()
            else:
                print("No results found")
            
            return 0
        
        # Legacy semantic search if requested
        if args.semantic_search:
            if not embedding_model:
                print("Error: Embedding model required for semantic search")
                return 1
            
            print(f"Performing semantic search for: '{args.semantic_search}'")
            results = enhanced_parser.search_emails_semantic(args.semantic_search, limit=10)
            
            print(f"Found {len(results)} semantically similar emails:")
            for i, email in enumerate(results, 1):
                similarity = email.get('similarity_score', 0)
                print(f"{i}. {email.get('subject', 'No subject')} (similarity: {similarity:.3f})")
                print(f"   From: {email.get('sender', 'Unknown')}")
                print(f"   Folder: {email.get('folder', 'Unknown')}")
                print()
            
            return 0
        
        # Show available options
        print("Enhanced Maildir Parser with Database Integration")
        print("=" * 60)
        print("Available options:")
        print("  --process-all: Process all folders to database (requires Maildir path)")
        print("  --folder FOLDER: Process specific folder only (requires Maildir path)")
        print("  --recompute-embeddings: Recompute all embeddings in database")
        print("  --batch-size N: Batch size for embedding recomputation (default: 100)")
        print("  --search QUERY: Search emails (SQL + semantic) - NO Maildir path needed!")
        print("  --search-type TYPE: Choose search type (sql/semantic/both)")
        print("  --search-fields FIELDS: Specify SQL search fields")
        print("  --case-sensitive: Perform case-sensitive SQL search")
        print("  --show-sql: Display the SQL query being executed")
        print("  --semantic-search QUERY: Legacy semantic search")
        print("  --no-embeddings: Skip computing embeddings")
        
        # Show database stats
        db_stats = enhanced_parser.get_database_stats()
        if db_stats.get('database_connected'):
            print(f"\nDatabase contains {db_stats.get('total_emails', 0)} emails")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        if 'db_manager' in locals():
            db_manager.disconnect()


if __name__ == "__main__":
    exit(main())
