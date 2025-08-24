#!/usr/bin/env python3
"""
Database Searcher for Email Archive

This module provides standalone search functionality for emails stored in the database,
without requiring access to the original Maildir folders.
"""

from typing import Dict, List, Any, Optional
from database_manager import DatabaseManager
from sentence_transformers import SentenceTransformer
from config import Config
import numpy as np
from datetime import datetime
from reranker import EmailReranker


class DatabaseSearcher:
    """
    Standalone search interface for emails stored in the database.
    No Maildir access required.
    """
    
    def __init__(self, db_manager: DatabaseManager, embedding_model: Optional[SentenceTransformer] = None, 
                 reranker: Optional[EmailReranker] = None):
        """
        Initialize the database searcher.
        
        Args:
            db_manager: Database manager instance
            embedding_model: Optional embedding model for semantic search
            reranker: Optional cross-encoder reranker for improving result relevance
        """
        self.db_manager = db_manager
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.semantic_depth_multiplier = 100  # Default: 100x the requested limit
    
    def search_emails_sql(self, search_term: str, search_fields: List[str] = None,
                         limit: int = 100, folder: str = None) -> List[Dict[str, Any]]:
        """
        Search emails using SQL LIKE (no Maildir required).
        
        Args:
            search_term: Text to search for
            search_fields: Fields to search in (default: ['subject', 'body', 'sender'])
            limit: Maximum number of results
            folder: Optional folder to limit search to
            
        Returns:
            List of matching emails
        """
        return self.db_manager.search_emails_sql_like(
            search_term, search_fields, limit, folder
        )
    
    def search_emails_semantic(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search emails using semantic similarity (no Maildir required).
        
        Args:
            query: Search query text
            limit: Maximum number of results
            
        Returns:
            List of matching emails with similarity scores
        """
        if not self.embedding_model:
            print("Warning: No embedding model available for semantic search")
            return []
        
        try:
            # Compute query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Use pgvector search directly from database manager
            # Use configurable semantic depth for better search coverage
            semantic_search_limit = min(limit * self.semantic_depth_multiplier, 10000)
            results = self.db_manager.search_emails_semantic(query, semantic_search_limit, query_embedding)
            
            # Add search_type to each result
            for result in results:
                result['search_type'] = 'semantic'
            
            # Apply cross-encoder reranking if available
            if self.reranker:
                print(f"🔄 Applying cross-encoder reranking...")
                results = self.reranker.rerank_results(query, results, top_k=limit)
                print(f"✅ Reranking complete!")
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def search_emails_hybrid(self, query: str, search_type: str = 'both',
                            search_fields: List[str] = None, limit: int = 100,
                            folder: str = None, case_sensitive: bool = False, show_sql: bool = False) -> Dict[str, Any]:
        """
        Hybrid search combining SQL LIKE and semantic search (no Maildir required).
        
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
        # For pure semantic search, use our optimized method
        if search_type == 'semantic' and self.embedding_model:
            semantic_results = self.search_emails_semantic(query, limit)
            return {
                'semantic_results': semantic_results,
                'sql_results': [],
                'combined_results': semantic_results,
                'search_metadata': {
                    'search_term': query,
                    'search_type': search_type,
                    'folder': folder,
                    'total_results': len(semantic_results),
                    'semantic_count': len(semantic_results),
                    'sql_count': 0
                }
            }
        
        # For other search types, use the database manager
        results = self.db_manager.search_emails_hybrid(
            query, search_type, search_fields, limit, folder, 0.1, case_sensitive, show_sql
        )
        
        # If semantic search is requested and we have an embedding model,
        # recompute semantic results with proper embeddings to get similarity scores
        if search_type in ['semantic', 'both'] and self.embedding_model and results.get('semantic_results'):
            # Compute query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Get fresh semantic results with similarity scores
            # Use a higher limit for semantic search to get better reranking candidates
            semantic_search_limit = min(limit * self.semantic_depth_multiplier, 10000)  # Configurable depth
            fresh_semantic_results = self.db_manager.search_emails_semantic(query, semantic_search_limit, query_embedding)
            
            # Update the results with fresh semantic results that have similarity scores
            results['semantic_results'] = fresh_semantic_results
            
            # Apply cross-encoder reranking to semantic results if available
            if self.reranker:
                print(f"🔄 Applying cross-encoder reranking to semantic results...")
                reranked_semantic = self.reranker.rerank_results(query, fresh_semantic_results, top_k=limit)
                results['semantic_results'] = reranked_semantic
                print(f"✅ Reranking complete!")
            
            # Update combined results if this is a hybrid search
            if search_type == 'both' and results.get('combined_results'):
                # Rebuild combined results with fresh semantic results
                combined = []
                seen_ids = set()
                semantic_ids = set()
                sql_ids = set()
                
                # Track semantic result IDs
                for email in fresh_semantic_results:
                    semantic_ids.add(email['id'])
                
                # Track SQL result IDs
                for email in results.get('sql_results', []):
                    sql_ids.add(email['id'])
                
                # Add fresh semantic results first (already sorted by cross-encoder)
                for email in fresh_semantic_results:
                    if email['id'] not in seen_ids:
                        email_copy = dict(email)
                        # Check if this email was found by both methods
                        if email['id'] in sql_ids:
                            email_copy['search_type'] = 'both'
                        else:
                            email_copy['search_type'] = 'semantic'
                        combined.append(email_copy)
                        seen_ids.add(email['id'])
                
                # Add SQL results (avoiding duplicates)
                for email in results.get('sql_results', []):
                    if email['id'] not in seen_ids:
                        email_copy = dict(email)
                        # Check if this email was found by both methods
                        if email['id'] in semantic_ids:
                            email_copy['search_type'] = 'both'
                        else:
                            email_copy['search_type'] = 'sql'
                        combined.append(email_copy)
                        seen_ids.add(email['id'])
                
                # Sort combined results by relevance (both first, then semantic, then by cross-encoder score, then by date)
                combined.sort(key=lambda x: (
                    x.get('search_type') == 'both',  # Both results first (highest priority)
                    x.get('search_type') == 'semantic',  # Then semantic results
                    x.get('cross_encoder_score', 0) if x.get('search_type') == 'semantic' else 0,  # Then by cross-encoder score for semantic
                    x.get('date_sent') or datetime.min     # Then by date
                ), reverse=True)
                
                # Ensure we have a balanced mix of results
                # Take up to limit/2 from each type, then fill remaining slots
                balanced_results = []
                semantic_count = 0
                sql_count = 0
                both_count = 0
                
                for email in combined:
                    if len(balanced_results) >= limit:
                        break
                    
                    search_type = email.get('search_type', 'unknown')
                    if search_type == 'both':
                        balanced_results.append(email)
                        both_count += 1
                    elif search_type == 'semantic' and semantic_count < limit // 2:
                        balanced_results.append(email)
                        semantic_count += 1
                    elif search_type == 'sql' and sql_count < limit // 2:
                        balanced_results.append(email)
                        sql_count += 1
                
                # Fill remaining slots with best remaining results
                for email in combined:
                    if len(balanced_results) >= limit:
                        break
                    if email not in balanced_results:
                        balanced_results.append(email)
                
                results['combined_results'] = balanced_results
                results['search_metadata']['total_results'] = len(balanced_results)
                results['search_metadata']['semantic_count'] = len(fresh_semantic_results)
        
        # Add search_type to semantic results if they exist
        if results.get('semantic_results') and self.embedding_model:
            for result in results['semantic_results']:
                result['search_type'] = 'semantic'
        
        # Add search_type to SQL results if they exist
        if results.get('sql_results'):
            for result in results['sql_results']:
                result['search_type'] = 'sql'
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics (no Maildir required).
        
        Returns:
            Dictionary with database statistics
        """
        try:
            email_count = self.db_manager.get_email_count()
            embedding_count = self.db_manager.get_embedding_count()
            
            return {
                'total_emails': email_count,
                'total_embeddings': embedding_count,
                'coverage_percentage': (embedding_count / email_count * 100) if email_count > 0 else 0
            }
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def get_folders(self) -> List[str]:
        """
        Get list of available folders in the database (no Maildir required).
        
        Returns:
            List of folder names
        """
        if not self.db_manager.connection:
            return []
        
        try:
            with self.db_manager.connection.cursor() as cursor:
                cursor.execute("SELECT DISTINCT folder FROM emails ORDER BY folder")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting folders: {e}")
            return []
    
    def get_emails_by_folder(self, folder: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get emails from a specific folder (no Maildir required).
        
        Args:
            folder: Folder name
            limit: Maximum number of results
            
        Returns:
            List of emails
        """
        return self.db_manager.get_emails_by_folder(folder, limit)
    
    def get_email_by_id(self, email_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific email by ID (no Maildir required).
        
        Args:
            email_id: Email ID in the database
            
        Returns:
            Email data or None if not found
        """
        if not self.db_manager.connection:
            return None
        
        try:
            with self.db_manager.connection.cursor(cursor_factory=self.db_manager.RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM emails WHERE id = %s", (email_id,))
                result = cursor.fetchone()
                return dict(result) if result else None
        except Exception as e:
            print(f"Error getting email by ID: {e}")
            return None
    
    def set_semantic_depth(self, multiplier: int):
        """
        Set the semantic search depth multiplier.
        
        Args:
            multiplier: How many times the requested limit to search (1-1000)
        """
        self.semantic_depth_multiplier = max(1, min(multiplier, 1000))
    
    def get_semantic_depth(self) -> int:
        """
        Get the current semantic search depth multiplier.
        
        Returns:
            Current depth multiplier
        """
        return self.semantic_depth_multiplier


def create_embedding_model():
    """Create and return the embedding model."""
    try:
        return SentenceTransformer('intfloat/e5-base')
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None


def main():
    """Command line interface for standalone database search."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Email Searcher (No Maildir Required)')
    
    # Database options (optional if config file exists)
    parser.add_argument('--db-host', help='PostgreSQL database host (overrides config)')
    parser.add_argument('--db-port', type=int, help='PostgreSQL database port (overrides config)')
    parser.add_argument('--db-name', help='PostgreSQL database name (overrides config)')
    parser.add_argument('--db-user', help='PostgreSQL database user (overrides config)')
    parser.add_argument('--db-password', help='PostgreSQL database password (overrides config)')
    
    # Search options
    parser.add_argument('--search', required=True, help='Search query')
    parser.add_argument('--search-type', choices=['sql', 'semantic', 'both'], default='both',
                       help='Search type: sql (LIKE), semantic (embeddings), or both (default: both)')
    parser.add_argument('--search-fields', nargs='+', 
                       default=['subject', 'body', 'sender'],
                       help='Fields to search in for SQL search (default: subject body sender)')
    parser.add_argument('--folder', help='Limit search to specific folder')
    parser.add_argument('--limit', type=int, default=20, help='Maximum results to return')
    parser.add_argument('--case-sensitive', action='store_true',
                       help='Perform case-sensitive SQL search (default: case-insensitive)')
    parser.add_argument('--show-sql', action='store_true',
                       help='Display the SQL query being executed')
    
    # Other options
    parser.add_argument('--list-folders', action='store_true', help='List available folders')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    
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
        # Initialize database manager
        db_manager = DatabaseManager(db_params)
        if not db_manager.connect():
            print("Failed to connect to database")
            return 1
        
        # Initialize embedding model if semantic search requested
        embedding_model = None
        if args.search_type in ['semantic', 'both']:
            embedding_model = create_embedding_model()
            if not embedding_model:
                print("Warning: Could not load embedding model. Semantic search will be disabled.")
                if args.search_type == 'semantic':
                    args.search_type = 'sql'
                elif args.search_type == 'both':
                    args.search_type = 'sql'
        
        # Initialize searcher
        searcher = DatabaseSearcher(db_manager, embedding_model)
        
        # List folders if requested
        if args.list_folders:
            folders = searcher.get_folders()
            print("Available folders in database:")
            for folder in folders:
                print(f"  - {folder}")
            print()
        
        # Show stats if requested
        if args.stats:
            stats = searcher.get_database_stats()
            print("Database Statistics:")
            print(f"  Total emails: {stats.get('total_emails', 0)}")
            print(f"  Total embeddings: {stats.get('total_embeddings', 0)}")
            print(f"  Coverage: {stats.get('coverage_percentage', 0):.1f}%")
            print()
        
        # Perform search
        print(f"Performing {args.search_type} search for: '{args.search}'")
        if args.folder:
            print(f"Folder: {args.folder}")
        
        results = searcher.search_emails_hybrid(
            query=args.search,
            search_type=args.search_type,
            search_fields=args.search_fields,
            limit=args.limit,
            folder=args.folder,
            case_sensitive=args.case_sensitive,
            show_sql=args.show_sql
        )
        
        if not results:
            print("No search results found")
            return 0
        
        # Display results
        metadata = results['search_metadata']
        print(f"\nSearch Results:")
        print(f"  Search Type: {metadata['search_type']}")
        print(f"  Folder: {metadata.get('folder', 'All folders')}")
        print(f"  SQL Results: {metadata.get('sql_count', 0)}")
        print(f"  Semantic Results: {metadata.get('semantic_count', 0)}")
        print(f"  Total Results: {metadata['total_results']}")
        
        # Display results
        if metadata['search_type'] == 'both' and results['combined_results']:
            emails_to_show = results['combined_results']
        elif metadata['search_type'] == 'sql' and results['sql_results']:
            emails_to_show = results['sql_results']
        elif metadata['search_type'] == 'semantic' and results['semantic_results']:
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
        
        # Clean up
        db_manager.disconnect()
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
