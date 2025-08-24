#!/usr/bin/env python3
"""
Database Manager for Maildir Email Parser

Handles PostgreSQL database operations including email storage and embeddings.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import re
from typing import Dict, Any, Optional, List


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self, connection_params: Dict[str, str]):
        """
        Initialize database manager.
        
        Args:
            connection_params: Dictionary with database connection parameters
        """
        self.connection_params = connection_params
        self.connection = None
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def create_tables(self) -> bool:
        """Create necessary database tables if they don't exist."""
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            with self.connection.cursor() as cursor:
                # Create emails table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS emails (
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
                    )
                """)
                
                # Create embeddings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS email_embeddings (
                        id SERIAL PRIMARY KEY,
                        email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
                        embedding_vector REAL[],
                        embedding_model VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_message_id ON emails(message_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_folder ON emails(folder)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_date_sent ON emails(date_sent)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_sender ON emails(sender)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_emails_subject ON emails(subject)")
                
                self.connection.commit()
                return True
                
        except Exception as e:
            print(f"Error creating tables: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def email_exists(self, message_id: str) -> bool:
        """Check if an email already exists in the database."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM emails WHERE message_id = %s", (message_id,))
                return cursor.fetchone() is not None
        except Exception as e:
            print(f"Error checking email existence: {e}")
            return False
    
    def save_email(self, email_data: Dict[str, Any]) -> Optional[int]:
        """
        Save email data to database.
        
        Args:
            email_data: Parsed email dictionary
            
        Returns:
            Database ID of the saved email, or None if failed
        """
        if not self.connection:
            if not self.connect():
                return None
        
        try:
            with self.connection.cursor() as cursor:
                # Parse date if available
                date_sent = None
                if email_data.get('date'):
                    try:
                        # Try to parse various date formats
                        date_str = email_data['date']
                        # Remove timezone info for simpler parsing
                        date_str = re.sub(r'\s*[+-]\d{4}', '', date_str)
                        date_sent = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S')
                    except:
                        pass
                
                cursor.execute("""
                    INSERT INTO emails (
                        message_id, subject, sender, recipient, date_sent, 
                        folder, content_type, body, headers, attachments
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    email_data.get('message_id', ''),
                    email_data.get('subject', ''),
                    email_data.get('from', ''),
                    email_data.get('to', ''),
                    date_sent,
                    email_data.get('folder', ''),
                    email_data.get('content_type', ''),
                    email_data.get('body', ''),
                    json.dumps(email_data.get('headers', {})),
                    json.dumps(email_data.get('attachments', []))
                ))
                
                email_id = cursor.fetchone()[0]
                self.connection.commit()
                return email_id
                
        except Exception as e:
            print(f"Error saving email: {e}")
            if self.connection:
                self.connection.rollback()
            return None
    
    def save_embedding(self, email_id: int, embedding: List[float], model_name: str) -> bool:
        """
        Save email embedding to database.
        
        Args:
            email_id: Database ID of the email
            embedding: List of embedding values
            model_name: Name of the embedding model used
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO email_embeddings (email_id, embedding_vector, embedding_model)
                    VALUES (%s, %s, %s)
                """, (email_id, embedding, model_name))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            print(f"Error saving embedding: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def get_email_count(self) -> int:
        """Get total number of emails in database."""
        if not self.connection:
            return 0
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM emails")
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error getting email count: {e}")
            return 0
    
    def search_emails_semantic(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search emails using semantic similarity with embeddings.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            List of emails with similarity scores
        """
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT e.*, 
                           eae.embedding_vector,
                           eae.embedding_model
                    FROM emails e
                    JOIN email_embeddings eae ON e.id = eae.email_id
                    WHERE eae.embedding_model = 'e5-base'
                    LIMIT %s
                """, (limit,))
                
                return cursor.fetchall()
                
        except Exception as e:
            print(f"Error searching emails semantically: {e}")
            return []
    
    def get_emails_by_folder(self, folder: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get emails from a specific folder.
        
        Args:
            folder: Folder name to search
            limit: Maximum number of results
            
        Returns:
            List of emails
        """
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM emails 
                    WHERE folder = %s 
                    ORDER BY date_sent DESC 
                    LIMIT %s
                """, (folder, limit))
                
                return cursor.fetchall()
                
        except Exception as e:
            print(f"Error getting emails by folder: {e}")
            return []
    
    def search_emails_sql_like(self, search_term: str, search_fields: List[str] = None, 
                              limit: int = 100, folder: str = None, case_sensitive: bool = False,
                              show_sql: bool = False) -> List[Dict[str, Any]]:
        """
        Search emails using SQL LIKE for traditional text-based search.
        
        Args:
            search_term: Text to search for
            search_fields: List of fields to search in (default: ['subject', 'body', 'sender'])
            limit: Maximum number of results to return
            folder: Optional folder to limit search to
            case_sensitive: Whether to perform case-sensitive search (default: False = case-insensitive)
            show_sql: Whether to display the SQL query being executed
            
        Returns:
            List of emails matching the search criteria
        """
        if not self.connection:
            return []
        
        if search_fields is None:
            search_fields = ['subject', 'body', 'sender']
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build dynamic WHERE clause based on search fields
                where_conditions = []
                params = []
                
                # Choose between LIKE (case-sensitive) and ILIKE (case-insensitive)
                like_operator = "LIKE" if case_sensitive else "ILIKE"
                
                for field in search_fields:
                    if field in ['subject', 'body', 'sender', 'recipient']:
                        where_conditions.append(f"{field} {like_operator} %s")
                        params.append(f"%{search_term}%")
                
                # Add folder filter if specified (case-insensitive for folder names)
                if folder:
                    where_conditions.append("folder ILIKE %s")
                    params.append(folder)
                
                # Build the complete query
                where_clause = " OR ".join(where_conditions) if where_conditions else "1=1"
                
                query = f"""
                    SELECT * FROM emails 
                    WHERE {where_clause}
                    ORDER BY date_sent DESC 
                    LIMIT %s
                """
                
                params.append(limit)
                
                # Display SQL query if requested
                if show_sql:
                    print("\n🔍 SQL Query:")
                    print("=" * 50)
                    print(query)
                    print("Parameters:", params)
                    print("=" * 50)
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except Exception as e:
            print(f"Error searching emails with SQL LIKE: {e}")
            return []
    
    def search_emails_hybrid(self, search_term: str, search_type: str = 'both',
                            search_fields: List[str] = None, limit: int = 100,
                            folder: str = None, similarity_threshold: float = 0.1,
                            case_sensitive: bool = False, show_sql: bool = False) -> Dict[str, Any]:
        """
        Hybrid search combining SQL LIKE and semantic search.
        
        Args:
            search_term: Text to search for
            search_type: 'sql', 'semantic', or 'both'
            search_fields: Fields for SQL search (default: ['subject', 'body', 'sender'])
            limit: Maximum results per search type
            folder: Optional folder to limit search to
            similarity_threshold: Minimum similarity for semantic search
            case_sensitive: Whether to perform case-sensitive SQL search (default: False)
            show_sql: Whether to display the SQL query being executed
            
        Returns:
            Dictionary with search results and metadata
        """
        results = {
            'sql_results': [],
            'semantic_results': [],
            'combined_results': [],
            'search_metadata': {
                'search_term': search_term,
                'search_type': search_type,
                'folder': folder,
                'total_results': 0
            }
        }
        
        # SQL LIKE search
        if search_type in ['sql', 'both']:
            sql_results = self.search_emails_sql_like(
                search_term, search_fields, limit, folder, case_sensitive, show_sql
            )
            results['sql_results'] = sql_results
            results['search_metadata']['sql_count'] = len(sql_results)
        
        # Semantic search (if embeddings exist)
        if search_type in ['semantic', 'both']:
            # Check if we have embeddings
            embedding_count = self.get_embedding_count()
            if embedding_count > 0:
                semantic_results = self.search_emails_semantic(search_term, limit)
                results['semantic_results'] = semantic_results
                results['search_metadata']['semantic_count'] = len(semantic_results)
            else:
                results['search_metadata']['semantic_count'] = 0
                results['search_metadata']['semantic_note'] = 'No embeddings available'
        
        # Combine results if both search types requested
        if search_type == 'both':
            combined = []
            seen_ids = set()
            
            # Calculate how many results to show from each type
            # Show up to limit/2 from each type, but ensure we show both types
            max_per_type = max(1, limit // 2)
            
            # Add semantic results first (up to max_per_type)
            semantic_added = 0
            for email in results['semantic_results']:
                if email['id'] not in seen_ids and semantic_added < max_per_type:
                    email_copy = dict(email)
                    email_copy['search_type'] = 'semantic'
                    combined.append(email_copy)
                    seen_ids.add(email['id'])
                    semantic_added += 1
            
            # Add SQL results (up to max_per_type, avoiding duplicates)
            sql_added = 0
            for email in results['sql_results']:
                if email['id'] not in seen_ids and sql_added < max_per_type:
                    email_copy = dict(email)
                    email_copy['search_type'] = 'sql'
                    combined.append(email_copy)
                    seen_ids.add(email['id'])
                    sql_added += 1
            
            # Sort combined results by relevance (semantic first, then by date)
            combined.sort(key=lambda x: (
                x.get('search_type') == 'semantic',  # Semantic results first
                x.get('date_sent') or datetime.min     # Then by date
            ), reverse=True)
            
            # Limit to requested limit
            results['combined_results'] = combined[:limit]
            results['search_metadata']['total_results'] = len(results['combined_results'])
        else:
            # Single search type
            if search_type == 'sql':
                results['search_metadata']['total_results'] = len(results['sql_results'])
            else:
                results['search_metadata']['total_results'] = len(results['semantic_results'])
        
        return results
    
    def get_embedding_count(self) -> int:
        """Get total number of embeddings in database."""
        if not self.connection:
            return 0
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM email_embeddings")
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error getting embedding count: {e}")
            return 0
