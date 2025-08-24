#!/usr/bin/env python3
"""
Enhanced Maildir Email Parser with Database Integration

This script extends the basic MaildirParser with PostgreSQL database integration
and sentence embeddings using sentence-transformers.
"""

from main import MaildirParser, html_to_plain_text
from database_manager import DatabaseManager
import numpy as np
from typing import List, Dict, Any, Optional


class EnhancedMaildirParser(MaildirParser):
    """Enhanced MaildirParser with database integration and embeddings."""
    
    def __init__(self, maildir_path: str, convert_html: bool = True, aggressive_clean: bool = False, 
                 db_manager: Optional[DatabaseManager] = None, embedding_model=None):
        """
        Initialize the EnhancedMaildirParser.
        
        Args:
            maildir_path: Path to the Maildir folder (root directory containing subdirectories)
            convert_html: Whether to convert HTML content to plain text (default: True)
            aggressive_clean: Whether to use aggressive CSS/HTML cleaning (default: False)
            db_manager: Optional database manager for saving emails
            embedding_model: Optional sentence transformer model for embeddings
        """
        super().__init__(maildir_path, convert_html, aggressive_clean)
        self.db_manager = db_manager
        self.embedding_model = embedding_model
    
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
                            if self.db_manager.save_embedding(email_id, embedding, 'all-MiniLM-L6-v2'):
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


def create_embedding_model():
    """
    Create and return a sentence transformer model.
    
    Returns:
        SentenceTransformer model or None if failed
    """
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence transformer model 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
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
    parser.add_argument('maildir_path', help='Path to the Maildir root folder')
    
    # Database options
    parser.add_argument('--db-host', required=True, help='PostgreSQL database host')
    parser.add_argument('--db-port', type=int, default=5432, help='PostgreSQL database port')
    parser.add_argument('--db-name', required=True, help='PostgreSQL database name')
    parser.add_argument('--db-user', required=True, help='PostgreSQL database user')
    parser.add_argument('--db-password', required=True, help='PostgreSQL database password')
    
    # Processing options
    parser.add_argument('--process-all', action='store_true', help='Process all folders to database')
    parser.add_argument('--folder', help='Process specific folder only')
    parser.add_argument('--no-embeddings', action='store_true', help='Skip computing embeddings')
    parser.add_argument('--semantic-search', help='Perform semantic search with query')
    
    args = parser.parse_args()
    
    try:
        # Initialize database manager
        db_params = {
            'host': args.db_host,
            'port': args.db_port,
            'database': args.db_name,
            'user': args.db_user,
            'password': args.db_password
        }
        
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
        enhanced_parser = EnhancedMaildirParser(
            args.maildir_path,
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
        
        # Semantic search if requested
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
        print("  --process-all: Process all folders to database")
        print("  --folder FOLDER: Process specific folder only")
        print("  --semantic-search QUERY: Search emails semantically")
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
