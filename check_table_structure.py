#!/usr/bin/env python3
"""
Check the current table structure to see what type embedding_vector column has.
"""

from database_manager import DatabaseManager
from config import Config

def main():
    # Load configuration
    config = Config()
    db_params = config.get_db_config()
    
    # Connect to database
    db_manager = DatabaseManager(db_params)
    if not db_manager.connect():
        print("Failed to connect to database")
        return 1
    
    try:
        with db_manager.connection.cursor() as cursor:
            # Check table structure
            cursor.execute("""
                SELECT column_name, data_type, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'email_embeddings' 
                AND column_name = 'embedding_vector'
            """)
            
            result = cursor.fetchone()
            if result:
                column_name, data_type, udt_name = result
                print(f"Column: {column_name}")
                print(f"Data Type: {data_type}")
                print(f"UDT Name: {udt_name}")
                
                # Check if pgvector extension exists
                cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                pgvector_exists = cursor.fetchone() is not None
                print(f"pgvector extension exists: {pgvector_exists}")
                
                # Check current table constraints
                cursor.execute("""
                    SELECT constraint_name, constraint_type 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'email_embeddings'
                """)
                
                constraints = cursor.fetchall()
                print(f"Table constraints: {constraints}")
                
            else:
                print("embedding_vector column not found")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        db_manager.disconnect()

if __name__ == "__main__":
    exit(main())
