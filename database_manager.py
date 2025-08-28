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
                
                # Create PDF documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pdf_documents (
                        id SERIAL PRIMARY KEY,
                        file_path VARCHAR(1000) UNIQUE,
                        file_name VARCHAR(500),
                        folder_name VARCHAR(100),
                        content TEXT,
                        content_length INTEGER,
                        page_count INTEGER,
                        file_size BIGINT,
                        title VARCHAR(500),
                        author VARCHAR(500),
                        subject VARCHAR(500),
                        creator VARCHAR(200),
                        producer VARCHAR(200),
                        created_date TIMESTAMP,
                        modified_date TIMESTAMP,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create PDF embeddings table
                if self.pgvector_available:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS pdf_embeddings (
                            id SERIAL PRIMARY KEY,
                            pdf_id INTEGER UNIQUE REFERENCES pdf_documents(id) ON DELETE CASCADE,
                            embedding_vector vector(768),
                            embedding_model VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                else:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS pdf_embeddings (
                            id SERIAL PRIMARY KEY,
                            pdf_id INTEGER UNIQUE REFERENCES pdf_documents(id) ON DELETE CASCADE,
                            embedding_vector REAL[],
                            embedding_model VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                
                # Create PDF indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pdf_documents_file_path ON pdf_documents(file_path)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pdf_documents_folder_name ON pdf_documents(folder_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pdf_documents_file_name ON pdf_documents(file_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pdf_documents_content_length ON pdf_documents(content_length)")
                
                # Create vector similarity index for PDFs (only if pgvector available)
                if self.pgvector_available:
                    try:
                        cursor.execute("""
                            CREATE INDEX IF NOT EXISTS idx_pdf_embeddings_vector 
                            ON pdf_embeddings 
                            USING ivfflat (embedding_vector vector_cosine_ops)
                            WITH (lists = 100)
                        """)
                        print("✅ PDF vector similarity index created")
                    except Exception as e:
                        print(f"⚠️  PDF vector index creation failed: {e}")
                
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
                # The caller should pass query_embedding for pgvector search
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
    
    # ==================== PDF Document Methods ====================
    
    def pdf_document_exists(self, file_path: str) -> bool:
        """Check if a PDF document already exists in the database."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pdf_documents WHERE file_path = %s", (file_path,))
                return cursor.fetchone() is not None
        except Exception as e:
            print(f"Error checking PDF document existence: {e}")
            return False
    
    def save_pdf_document(self, file_path: str, file_name: str, folder_name: str, 
                         content: str, metadata: Dict[str, Any], embedding: List[float]) -> bool:
        """
        Save PDF document data to database.
        
        Args:
            file_path: Full path to the PDF file
            file_name: Name of the PDF file
            folder_name: Folder/category name
            content: Extracted text content
            metadata: PDF metadata dictionary
            embedding: Document embedding vector
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            with self.connection.cursor() as cursor:
                # Insert PDF document
                cursor.execute("""
                    INSERT INTO pdf_documents (
                        file_path, file_name, folder_name, content, content_length,
                        page_count, file_size, title, author, subject, creator, producer,
                        created_date, modified_date, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (file_path) DO UPDATE SET
                        file_name = EXCLUDED.file_name,
                        folder_name = EXCLUDED.folder_name,
                        content = EXCLUDED.content,
                        content_length = EXCLUDED.content_length,
                        page_count = EXCLUDED.page_count,
                        file_size = EXCLUDED.file_size,
                        title = EXCLUDED.title,
                        author = EXCLUDED.author,
                        subject = EXCLUDED.subject,
                        creator = EXCLUDED.creator,
                        producer = EXCLUDED.producer,
                        created_date = EXCLUDED.created_date,
                        modified_date = EXCLUDED.modified_date,
                        metadata = EXCLUDED.metadata,
                        created_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (
                    file_path, file_name, folder_name, content, metadata.get('content_length', 0),
                    metadata.get('page_count', 0), metadata.get('file_size', 0),
                    metadata.get('title', ''), metadata.get('author', ''),
                    metadata.get('subject', ''), metadata.get('creator', ''),
                    metadata.get('producer', ''), metadata.get('created_date'),
                    metadata.get('modified_date'), json.dumps(metadata)
                ))
                
                result = cursor.fetchone()
                if result:
                    pdf_id = result[0]
                    
                    # Save embedding
                    if self.save_pdf_embedding(pdf_id, embedding, 'e5-base'):
                        self.connection.commit()
                        return True
                    else:
                        self.connection.rollback()
                        return False
                else:
                    self.connection.rollback()
                    return False
                    
        except Exception as e:
            print(f"Error saving PDF document: {e}")
            if self.connection:
                self.connection.rollback()
            return False
    
    def save_pdf_embedding(self, pdf_id: int, embedding: List[float], model_name: str = 'e5-base') -> bool:
        """
        Save PDF document embedding to database.
        
        Args:
            pdf_id: Database ID of the PDF document
            embedding: Embedding vector
            model_name: Name of the embedding model
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                if self.pgvector_available:
                    cursor.execute("""
                        INSERT INTO pdf_embeddings (pdf_id, embedding_vector, embedding_model)
                        VALUES (%s, %s::vector, %s)
                        ON CONFLICT (pdf_id) DO UPDATE SET
                            embedding_vector = EXCLUDED.embedding_vector,
                            embedding_model = EXCLUDED.embedding_model,
                            created_at = CURRENT_TIMESTAMP
                    """, (pdf_id, embedding, model_name))
                else:
                    cursor.execute("""
                        INSERT INTO pdf_embeddings (pdf_id, embedding_vector, embedding_model)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (pdf_id) DO UPDATE SET
                            embedding_vector = EXCLUDED.embedding_vector,
                            embedding_model = EXCLUDED.embedding_model,
                            created_at = CURRENT_TIMESTAMP
                    """, (pdf_id, embedding, model_name))
                
                return True
                
        except Exception as e:
            print(f"Error saving PDF embedding: {e}")
            return False
    
    def search_pdf_documents(self, query: str, query_embedding: List[float], 
                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search PDF documents using semantic similarity.
        
        Args:
            query: Search query text
            query_embedding: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of matching PDF documents with similarity scores
        """
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # First check if we have any PDF documents with embeddings
                cursor.execute("""
                    SELECT COUNT(*) FROM pdf_documents pd
                    JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                    WHERE pe.embedding_vector IS NOT NULL
                """)
                result = cursor.fetchone()
                print(f"🔍 Debug: cursor result type: {type(result)}, value: {result}")
                
                if result is None:
                    print("🔍 No PDF documents with embeddings found in database")
                    return []
                
                # Handle different result types
                if isinstance(result, dict):
                    total_count = result.get('count', 0)
                elif isinstance(result, (tuple, list)):
                    total_count = result[0] if result else 0
                else:
                    total_count = int(result) if result else 0
                
                if total_count == 0:
                    print("🔍 No PDF documents with embeddings found in database")
                    return []
                
                print(f"🔍 Found {total_count} PDF documents with embeddings")
                

                
                embedding_array = np.array(query_embedding, dtype=np.float32)
                embedding_list = embedding_array.tolist()  # Convert to list for psycopg2
                
                # Use pgvector for fast similarity search
                cursor.execute("""
                    SELECT pd.id, pd.file_path, pd.file_name, pd.folder_name, pd.content, 
                           pd.content_length, pd.page_count, pd.file_size, pd.title, pd.author,
                           pd.subject, pd.creator, pd.producer, pd.created_date, pd.modified_date,
                           pd.metadata, pd.created_at as doc_created_at,
                           pe.embedding_model, pe.embedding_vector,
                           1 - (pe.embedding_vector <=> %s::vector) as similarity_score
                    FROM pdf_documents pd
                    JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                    WHERE pe.embedding_vector IS NOT NULL
                    ORDER BY 1 - (pe.embedding_vector <=> %s::vector) DESC
                    LIMIT %s
                """, (embedding_list, embedding_list, limit))
                
                # Return pgvector results directly
                results = cursor.fetchall()
                print(f"🔍 pgvector search for pdf documents returned {len(results)} results")
                
                # Convert string vectors back to proper vectors if needed
                for result in results:
                    if isinstance(result['embedding_vector'], str):
                        # The vector was returned as a string, convert it back
                        try:
                            # Parse the string representation back to a list
                            import ast
                            vector_str = result['embedding_vector'].strip()
                            if vector_str.startswith('[') and vector_str.endswith(']'):
                                vector_list = ast.literal_eval(vector_str)
                                result['embedding_vector'] = vector_list
                        except Exception as e:
                            print(f"Warning: Could not parse vector string: {e}")
                
                return results
                
        except Exception as e:
            print(f"Error searching PDF documents: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_pdf_documents_without_embeddings(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get PDF documents that don't have embeddings yet."""
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT pd.* FROM pdf_documents pd
                    LEFT JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                    WHERE pe.pdf_id IS NULL
                    ORDER BY pd.created_at DESC
                    LIMIT %s
                """, (limit,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error getting PDF documents without embeddings: {e}")
            return []
    
    def clear_all_pdf_embeddings(self) -> bool:
        """Clear all PDF embeddings from the database."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM pdf_embeddings")
                self.connection.commit()
                print(f"Cleared all PDF embeddings from database")
                return True
        except Exception as e:
            print(f"Error clearing PDF embeddings: {e}")
            return False

    # Admin methods for service administrators
    def get_all_emails(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all emails from the database (admin function)."""
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM emails 
                    ORDER BY date_sent DESC 
                    LIMIT %s
                """, (limit,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error getting all emails: {e}")
            return []

    def get_email_by_id(self, email_id: int) -> Optional[Dict[str, Any]]:
        """Get email by ID (admin function)."""
        if not self.connection:
            return None
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM emails WHERE id = %s", (email_id,))
                return cursor.fetchone()
        except Exception as e:
            print(f"Error getting email by ID: {e}")
            return None

    def update_email(self, email_id: int, update_data: Dict[str, Any]) -> bool:
        """Update email data (admin function)."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Build dynamic UPDATE query
                set_clauses = []
                values = []
                
                allowed_fields = ['subject', 'sender', 'recipient', 'folder', 'body', 'content_type']
                for field, value in update_data.items():
                    if field in allowed_fields:
                        set_clauses.append(f"{field} = %s")
                        values.append(value)
                
                if not set_clauses:
                    return False
                
                values.append(email_id)
                query = f"UPDATE emails SET {', '.join(set_clauses)} WHERE id = %s"
                
                cursor.execute(query, values)
                self.connection.commit()
                return True
        except Exception as e:
            print(f"Error updating email: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def delete_email(self, email_id: int) -> bool:
        """Delete email by ID (admin function)."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Delete related embeddings first
                cursor.execute("DELETE FROM email_embeddings WHERE email_id = %s", (email_id,))
                
                # Delete the email
                cursor.execute("DELETE FROM emails WHERE id = %s", (email_id,))
                
                self.connection.commit()
                return True
        except Exception as e:
            print(f"Error deleting email: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def get_all_pdf_documents(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all PDF documents from the database (admin function)."""
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM pdf_documents 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error getting all PDF documents: {e}")
            return []

    def get_pdf_document_by_id(self, pdf_id: int) -> Optional[Dict[str, Any]]:
        """Get PDF document by ID (admin function)."""
        if not self.connection:
            return None
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM pdf_documents WHERE id = %s", (pdf_id,))
                return cursor.fetchone()
        except Exception as e:
            print(f"Error getting PDF document by ID: {e}")
            return None

    def update_pdf_document(self, pdf_id: int, update_data: Dict[str, Any]) -> bool:
        """Update PDF document data (admin function)."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Build dynamic UPDATE query
                set_clauses = []
                values = []
                
                allowed_fields = ['file_name', 'folder_name', 'title', 'author', 'subject', 'content']
                for field, value in update_data.items():
                    if field in allowed_fields:
                        set_clauses.append(f"{field} = %s")
                        values.append(value)
                
                if not set_clauses:
                    return False
                
                values.append(pdf_id)
                query = f"UPDATE pdf_documents SET {', '.join(set_clauses)} WHERE id = %s"
                
                cursor.execute(query, values)
                self.connection.commit()
                return True
        except Exception as e:
            print(f"Error updating PDF document: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def delete_pdf_document(self, pdf_id: int) -> bool:
        """Delete PDF document by ID (admin function)."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Delete related embeddings first
                cursor.execute("DELETE FROM pdf_embeddings WHERE pdf_id = %s", (pdf_id,))
                
                # Delete the PDF document
                cursor.execute("DELETE FROM pdf_documents WHERE id = %s", (pdf_id,))
                
                self.connection.commit()
                return True
        except Exception as e:
            print(f"Error deleting PDF document: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def get_all_projects(self) -> List[Dict[str, Any]]:
        """Get all projects (placeholder for future project management)."""
        # This is a placeholder for future project management functionality
        return []

    def get_project_by_id(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get project by ID (placeholder for future project management)."""
        # This is a placeholder for future project management functionality
        return None

    def update_project(self, project_id: int, update_data: Dict[str, Any]) -> bool:
        """Update project data (placeholder for future project management)."""
        # This is a placeholder for future project management functionality
        return False

    def delete_project(self, project_id: int) -> bool:
        """Delete project by ID (placeholder for future project management)."""
        # This is a placeholder for future project management functionality
        return False

    def recompute_all_embeddings(self) -> bool:
        """Recompute all embeddings in the database (admin function)."""
        if not self.connection:
            return False
        
        try:
            # This would trigger a full recomputation of all embeddings
            # Implementation depends on your embedding recomputation logic
            print("Starting full embedding recomputation...")
            # TODO: Implement full recomputation logic
            return True
        except Exception as e:
            print(f"Error recomputing embeddings: {e}")
            return False

    def cleanup_orphaned_records(self) -> bool:
        """Clean up orphaned records (admin function)."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Clean up orphaned email embeddings
                cursor.execute("""
                    DELETE FROM email_embeddings 
                    WHERE email_id NOT IN (SELECT id FROM emails)
                """)
                orphaned_emails = cursor.rowcount
                
                # Clean up orphaned PDF embeddings
                cursor.execute("""
                    DELETE FROM pdf_embeddings 
                    WHERE pdf_id NOT IN (SELECT id FROM pdf_documents)
                """)
                orphaned_pdfs = cursor.rowcount
                
                self.connection.commit()
                print(f"Cleaned up {orphaned_emails} orphaned email embeddings and {orphaned_pdfs} orphaned PDF embeddings")
                return True
        except Exception as e:
            print(f"Error cleaning up orphaned records: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def optimize_tables(self) -> bool:
        """Optimize database tables (admin function)."""
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Analyze tables for better query planning
                cursor.execute("ANALYZE emails")
                cursor.execute("ANALYZE pdf_documents")
                cursor.execute("ANALYZE email_embeddings")
                cursor.execute("ANALYZE pdf_embeddings")
                
                # Vacuum tables to reclaim storage and update statistics
                cursor.execute("VACUUM ANALYZE emails")
                cursor.execute("VACUUM ANALYZE pdf_documents")
                cursor.execute("VACUUM ANALYZE email_embeddings")
                cursor.execute("VACUUM ANALYZE pdf_embeddings")
                
                self.connection.commit()
                print("Database tables optimized successfully")
                return True
        except Exception as e:
            print(f"Error optimizing tables: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    # GitHub Issues Search Methods
    
    def search_github_issues(self, query: str, query_embedding: List[float],
                            limit: int = 10, repository: Optional[str] = None,
                            state: Optional[str] = None, author: Optional[str] = None,
                            labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search GitHub issues using semantic similarity.
        
        Args:
            query: Search query text
            query_embedding: Query embedding vector
            limit: Maximum number of results
            repository: Optional repository filter
            state: Optional state filter
            author: Optional author filter
            labels: Optional list of label filters
            
        Returns:
            List of matching issues with similarity scores
        """
        if not self.connection:
            return []
        
        try:
            # Check if we have any GitHub issues with embeddings
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) FROM github_issues gi
                    JOIN github_issue_embeddings gie ON gi.issue_id = gie.issue_id
                    WHERE gie.embedding_vector IS NOT NULL
                """)
                result = cursor.fetchone()
                if result is None or result[0] == 0:
                    print("🔍 No GitHub issues with embeddings found")
                    return []
                
                total_count = result[0]
                print(f"🔍 Found {total_count} GitHub issues with embeddings")
            
            # Build the search query with filters
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build WHERE clause with filters
                where_conditions = ["gie.embedding_vector IS NOT NULL"]
                params = []
                
                if repository:
                    where_conditions.append("gi.repository = %s")
                    params.append(repository)
                
                if state and state != 'all':
                    where_conditions.append("gi.state = %s")
                    params.append(state)
                
                if author:
                    where_conditions.append("gi.author = %s")
                    params.append(author)
                
                if labels:
                    label_conditions = []
                    for label in labels:
                        label_conditions.append("gi.labels @> %s")
                        params.append([label])
                    where_conditions.append(f"({' OR '.join(label_conditions)})")
                
                where_clause = " AND ".join(where_conditions)
                
                # Add embedding vector and limit to params
                embedding_array = np.array(query_embedding, dtype=np.float32)
                embedding_list = embedding_array.tolist()
                params.extend([embedding_list, embedding_list, limit])
                
                cursor.execute(f"""
                    SELECT gi.issue_id, gi.repository, gi.title, gi.body, gi.state,
                           gi.labels, gi.assignees, gi.author, gi.created_at, gi.updated_at,
                           gi.closed_at, gi.comments_count, gi.html_url, gi.issue_number,
                           gi.milestone, gi.reactions, gi.created_at_db,
                           gie.embedding_model,
                           1 - (gie.embedding_vector <=> %s::vector) as similarity_score
                    FROM github_issues gi
                    JOIN github_issue_embeddings gie ON gi.issue_id = gie.issue_id
                    WHERE {where_clause}
                    ORDER BY 1 - (gie.embedding_vector <=> %s::vector) DESC
                    LIMIT %s
                """, params)
                
                results = cursor.fetchall()
                print(f"🔍 pgvector search for GitHub issues returned {len(results)} results")
                return results
                
        except Exception as e:
            print(f"Error in GitHub issues semantic search: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def search_github_issues_sql(self, query: str, limit: int = 10,
                                repository: Optional[str] = None,
                                state: Optional[str] = None,
                                author: Optional[str] = None,
                                labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search GitHub issues using SQL LIKE queries.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            repository: Optional repository filter
            state: Optional state filter
            author: Optional author filter
            labels: Optional list of label filters
            
        Returns:
            List of matching issues
        """
        if not self.connection:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build WHERE clause with filters
                where_conditions = []
                params = []
                
                # Text search conditions
                search_conditions = []
                for field in ['title', 'body']:
                    search_conditions.append(f"{field} ILIKE %s")
                    params.append(f"%{query}%")
                
                if search_conditions:
                    where_conditions.append(f"({' OR '.join(search_conditions)})")
                
                # Repository filter
                if repository:
                    where_conditions.append("repository = %s")
                    params.append(repository)
                
                # State filter
                if state and state != 'all':
                    where_conditions.append("state = %s")
                    params.append(state)
                
                # Author filter
                if author:
                    where_conditions.append("author = %s")
                    params.append(author)
                
                # Labels filter
                if labels:
                    label_conditions = []
                    for label in labels:
                        label_conditions.append("labels @> %s")
                        params.append([label])
                    where_conditions.append(f"({' OR '.join(label_conditions)})")
                
                # Build final query
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                params.append(limit)
                
                cursor.execute(f"""
                    SELECT issue_id, repository, title, body, state, labels, assignees,
                           author, created_at, updated_at, closed_at, comments_count,
                           html_url, issue_number, milestone, reactions, created_at_db
                    FROM github_issues
                    WHERE {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT %s
                """, params)
                
                return cursor.fetchall()
                
        except Exception as e:
            print(f"Error in GitHub issues SQL search: {e}")
            return []
    
    def store_github_issue_embedding(self, issue_id: int, embedding_model: str,
                                   embedding_vector: List[float]) -> bool:
        """
        Store embedding for a GitHub issue.
        
        Args:
            issue_id: GitHub issue ID
            embedding_model: Name of the embedding model
            embedding_vector: Embedding vector
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            return False
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO github_issue_embeddings (issue_id, embedding_model, embedding_vector)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (issue_id, embedding_model) DO UPDATE SET
                        embedding_vector = EXCLUDED.embedding_vector,
                        created_at = CURRENT_TIMESTAMP
                """, (issue_id, embedding_model, embedding_vector))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            print(f"Error storing GitHub issue embedding: {e}")
            if self.connection:
                self.connection.rollback()
            return False
