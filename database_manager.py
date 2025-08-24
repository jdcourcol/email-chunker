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
import numpy as np


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
            
            # Check if pgvector extension is available
            if not hasattr(self, 'pgvector_available'):
                try:
                    with self.connection.cursor() as cursor:
                        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                        self.pgvector_available = cursor.fetchone() is not None
                        if self.pgvector_available:
                            print(f"✅ pgvector extension detected - using vector similarity search")
                        else:
                            print(f"⚠️  pgvector extension not found - using in-memory similarity search")
                except Exception as e:
                    print(f"⚠️  Could not check pgvector extension: {e}")
                    self.pgvector_available = False
            
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
                
                # Try to create pgvector extension first
                self.pgvector_available = False
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    self.pgvector_available = True
                    print("✅ pgvector extension enabled - using vector similarity search")
                except Exception as e:
                    print(f"⚠️  pgvector extension not available: {e}")
                    print("   Falling back to in-memory similarity search")
                    self.pgvector_available = False
                
                # Create embeddings table based on pgvector availability
                if self.pgvector_available:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS email_embeddings (
                            id SERIAL PRIMARY KEY,
                            email_id INTEGER UNIQUE REFERENCES emails(id) ON DELETE CASCADE,
                            embedding_vector vector(768),  -- e5-base has 768 dimensions
                            embedding_model VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                else:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS email_embeddings (
                            id SERIAL PRIMARY KEY,
                            email_id INTEGER UNIQUE REFERENCES emails(id) ON DELETE CASCADE,
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
                
                # Create vector similarity index for fast semantic search (only if pgvector available)
                if self.pgvector_available:
                    try:
                        cursor.execute("""
                            CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
                            ON email_embeddings 
                            USING ivfflat (embedding_vector vector_cosine_ops)
                            WITH (lists = 100)
                        """)
                        print("✅ Vector similarity index created")
                    except Exception as e:
                        print(f"⚠️  Vector index creation failed: {e}")
                
                self.connection.commit()
                
                # Ensure constraints exist (for existing tables)
                self.ensure_embedding_constraints()
                
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
                # Convert embedding to list for database storage
                if isinstance(embedding, np.ndarray):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = list(embedding)
                
                cursor.execute("""
                    INSERT INTO email_embeddings (email_id, embedding_vector, embedding_model)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (email_id) DO UPDATE SET
                        embedding_vector = EXCLUDED.embedding_vector,
                        embedding_model = EXCLUDED.embedding_model,
                        created_at = CURRENT_TIMESTAMP
                """, (email_id, embedding_list, model_name))
                
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
    
    def search_emails_semantic(self, query: str, limit: int = 10, 
                              query_embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Search emails using semantic similarity with pgvector or fallback to in-memory.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            query_embedding: Pre-computed query embedding (optional)
            
        Returns:
            List of emails with similarity scores
        """
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                if query_embedding and hasattr(self, 'pgvector_available') and self.pgvector_available:
                    # Use pgvector cosine similarity search
                    print(f"DEBUG: Using pgvector search, pgvector_available={self.pgvector_available}")
                    try:
                        embedding_array = np.array(query_embedding, dtype=np.float32)
                        embedding_list = embedding_array.tolist()  # Convert to list for psycopg2
                        print(f"DEBUG: Query embedding shape: {embedding_array.shape}, dtype: {embedding_array.dtype}")
                        
                        cursor.execute("""
                            SELECT e.*, 
                                   eae.embedding_model,
                                   1 - (eae.embedding_vector <=> %s::vector) as similarity_score
                            FROM emails e
                            JOIN email_embeddings eae ON e.id = eae.email_id
                            WHERE eae.embedding_model = 'e5-base'
                            ORDER BY eae.embedding_vector <=> %s::vector
                            LIMIT %s
                        """, (embedding_list, embedding_list, limit))
                        
                        results = cursor.fetchall()
                        print(f"DEBUG: pgvector search returned {len(results)} results")
                        
                        # Convert similarity scores to float
                        for result in results:
                            if 'similarity_score' in result:
                                result['similarity_score'] = float(result['similarity_score'])
                        
                        return results
                        
                    except Exception as pgvector_error:
                        print(f"pgvector search failed, falling back to in-memory: {pgvector_error}")
                        import traceback
                        traceback.print_exc()
                        # Fall through to in-memory search
                else:
                    print(f"DEBUG: pgvector not available, pgvector_available={getattr(self, 'pgvector_available', 'NOT_SET')}")
                    print(f"DEBUG: hasattr pgvector_available={hasattr(self, 'pgvector_available')}")
                    print(f"DEBUG: query_embedding provided={query_embedding is not None}")
                
                # Fallback to in-memory similarity search
                cursor.execute("""
                    SELECT e.*, 
                           eae.embedding_vector,
                           eae.embedding_model
                    FROM emails e
                    JOIN email_embeddings eae ON e.id = eae.email_id
                    WHERE eae.embedding_model = 'e5-base'
                    LIMIT %s
                """, (limit * 2,))  # Get more results for better in-memory selection
                
                results = cursor.fetchall()
                
                if not results or not query_embedding:
                    return results
                
                # Compute similarity scores in memory
                query_embedding_array = np.array(query_embedding)
                scored_results = []
                
                for email in results:
                    if 'embedding_vector' in email and email['embedding_vector']:
                        try:
                            # Convert embedding to numpy array for similarity computation
                            email_embedding = np.array(email['embedding_vector'])
                            
                            # Compute cosine similarity
                            similarity = np.dot(email_embedding, query_embedding_array) / (
                                np.linalg.norm(email_embedding) * np.linalg.norm(query_embedding_array)
                            )
                            
                            # Add similarity score to email data
                            email_copy = dict(email)
                            email_copy['similarity_score'] = float(similarity)
                            scored_results.append(email_copy)
                        except Exception as e:
                            print(f"Error computing similarity for email {email.get('id', 'unknown')}: {e}")
                            continue
                
                # Sort by similarity score (descending) and return top results
                scored_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                return scored_results[:limit]
                
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
                # Note: query_embedding will be computed by the caller
                # For now, we'll use the basic search and let the caller handle embeddings
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
            
            # Update counts to show what was actually added to combined results
            results['search_metadata']['sql_count'] = sql_added
            results['search_metadata']['semantic_count'] = semantic_added
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
    
    def ensure_embedding_constraints(self) -> bool:
        """Ensure the email_embeddings table has the required constraints."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Check if unique constraint exists
                cursor.execute("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'email_embeddings' 
                    AND constraint_type = 'UNIQUE' 
                    AND constraint_name LIKE '%email_id%'
                """)
                
                if not cursor.fetchone():
                    print("Adding unique constraint to email_id column...")
                    cursor.execute("""
                        ALTER TABLE email_embeddings 
                        ADD CONSTRAINT email_embeddings_email_id_unique 
                        UNIQUE (email_id)
                    """)
                    self.connection.commit()
                    print("✅ Unique constraint added successfully")
                    return True
                else:
                    print("✅ Unique constraint already exists")
                    return True
                    
        except Exception as e:
            print(f"Error ensuring constraints: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def clear_all_embeddings(self) -> bool:
        """Clear all embeddings from the database."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM email_embeddings")
                self.connection.commit()
                print(f"Cleared all embeddings from database")
                return True
        except Exception as e:
            print(f"Error clearing embeddings: {e}")
            return False
    
    def get_emails_without_embeddings(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get emails that don't have embeddings yet."""
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT e.* FROM emails e
                    LEFT JOIN email_embeddings eae ON e.id = eae.email_id
                    WHERE eae.email_id IS NULL
                    ORDER BY e.date_sent DESC
                    LIMIT %s
                """, (limit,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error getting emails without embeddings: {e}")
            return []
    
    def recompute_embeddings(self, embedding_model, model_name: str = 'e5-base', 
                            batch_size: int = 100, show_progress: bool = True) -> Dict[str, int]:
        """
        Recompute embeddings for all emails in the database.
        
        Args:
            embedding_model: SentenceTransformer model instance
            model_name: Name of the embedding model
            batch_size: Number of emails to process in each batch
            show_progress: Whether to show progress updates
            
        Returns:
            Dictionary with recomputation statistics
        """
        if not self.connection:
            return {'processed': 0, 'embeddings_created': 0, 'errors': 0}
        
        try:
            # Clear existing embeddings
            if show_progress:
                print("Clearing existing embeddings...")
            self.clear_all_embeddings()
            
            # Get all emails
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM emails")
                result = cursor.fetchone()
                if result is None:
                    print("Error: COUNT query returned no results")
                    return {'processed': 0, 'embeddings_created': 0, 'errors': 0}
                total_emails = result[0]
            
            if total_emails == 0:
                print("No emails found in database")
                return {'processed': 0, 'embeddings_created': 0, 'errors': 0}
            
            if show_progress:
                print(f"Found {total_emails} emails to process")
            
            processed = 0
            embeddings_created = 0
            errors = 0
            
            # Process emails in batches
            offset = 0
            while offset < total_emails:
                with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT * FROM emails 
                        ORDER BY date_sent DESC 
                        LIMIT %s OFFSET %s
                    """, (batch_size, offset))
                    
                    batch = cursor.fetchall()
                    if not batch:
                        break
                    
                    for email in batch:
                        try:
                            # Create text for embedding (subject + body)
                            text_for_embedding = f"{email.get('subject', '')} {email.get('body', '')}"
                            text_for_embedding = text_for_embedding.strip()
                            
                            if text_for_embedding:
                                # Compute embedding
                                embedding = embedding_model.encode(text_for_embedding).tolist()
                                
                                # Save embedding to database
                                if self.save_embedding(email['id'], embedding, model_name):
                                    embeddings_created += 1
                                else:
                                    errors += 1
                            else:
                                errors += 1
                                
                        except Exception as e:
                            if show_progress:
                                print(f"Error processing email {email.get('id', 'unknown')}: {e}")
                            errors += 1
                        
                        processed += 1
                        
                        if show_progress and processed % 10 == 0:
                            print(f"Processed {processed}/{total_emails} emails...")
                
                offset += batch_size
            
            if show_progress:
                print(f"\nRecomputation complete!")
                print(f"  Processed: {processed}")
                print(f"  Embeddings created: {embeddings_created}")
                print(f"  Errors: {errors}")
            
            return {
                'processed': processed,
                'embeddings_created': embeddings_created,
                'errors': errors
            }
            
        except Exception as e:
            print(f"Error during recomputation: {e}")
            import traceback
            traceback.print_exc()
            return {'processed': 0, 'embeddings_created': 0, 'errors': 0}
