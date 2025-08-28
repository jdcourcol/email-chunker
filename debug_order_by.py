#!/usr/bin/env python3
"""
Debug the ORDER BY clause to find why it's not working.
"""

import sys
from config import Config
from database_manager import DatabaseManager

def debug_order_by():
    """Debug the ORDER BY clause."""
    print("🔍 Debugging ORDER BY Clause")
    print("=" * 30)
    
    try:
        # Initialize configuration
        config = Config()
        db_config = config.get_db_config()
        
        if not all(db_config.get(key) for key in ['host', 'database', 'user', 'password']):
            print("❌ Incomplete database configuration")
            return
        
        print(f"✅ Database config loaded")
        
        # Initialize database manager
        db_manager = DatabaseManager(db_config)
        if not db_manager.connect():
            print("❌ Failed to connect to database")
            return
        
        print("✅ Database connection established")
        
        test_vector = [0.1] * 768
        
        # Test 1: Check if the similarity operator works in ORDER BY
        print("\n🔧 Test 1: ORDER BY with similarity operator")
        with db_manager.connection.cursor() as cursor:
            try:
                cursor.execute("""
                    SELECT pe.pdf_id, pe.embedding_vector <=> %s::vector as distance
                    FROM pdf_embeddings pe
                    WHERE pe.embedding_vector IS NOT NULL
                    ORDER BY pe.embedding_vector <=> %s::vector
                    LIMIT 3
                """, (test_vector, test_vector))
                
                results = cursor.fetchall()
                print(f"   ✅ ORDER BY similarity returned {len(results)} results")
                for result in results:
                    print(f"     PDF ID {result[0]}: distance {result[1]:.6f}")
            except Exception as e:
                print(f"   ❌ ORDER BY similarity failed: {e}")
        
        # Test 2: Check if the issue is with the 1 - similarity calculation
        print("\n🔧 Test 2: ORDER BY with 1 - similarity calculation")
        with db_manager.connection.cursor() as cursor:
            try:
                cursor.execute("""
                    SELECT pe.pdf_id, 1 - (pe.embedding_vector <=> %s::vector) as similarity
                    FROM pdf_embeddings pe
                    WHERE pe.embedding_vector IS NOT NULL
                    ORDER BY 1 - (pe.embedding_vector <=> %s::vector) DESC
                    LIMIT 3
                """, (test_vector, test_vector))
                
                results = cursor.fetchall()
                print(f"   ✅ ORDER BY 1-similarity returned {len(results)} results")
                for result in results:
                    print(f"     PDF ID {result[0]}: similarity {result[1]:.6f}")
            except Exception as e:
                print(f"   ❌ ORDER BY 1-similarity failed: {e}")
        
        # Test 3: Check if the issue is with the JOIN
        print("\n🔧 Test 3: ORDER BY with JOIN")
        with db_manager.connection.cursor() as cursor:
            try:
                cursor.execute("""
                    SELECT pd.id, pd.file_name, pe.embedding_vector <=> %s::vector as distance
                    FROM pdf_documents pd
                    JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                    WHERE pe.embedding_vector IS NOT NULL
                    ORDER BY pe.embedding_vector <=> %s::vector
                    LIMIT 3
                """, (test_vector, test_vector))
                
                results = cursor.fetchall()
                print(f"   ✅ ORDER BY with JOIN returned {len(results)} results")
                for result in results:
                    print(f"     ID {result[0]}: {result[1]}, distance {result[2]:.6f}")
            except Exception as e:
                print(f"   ❌ ORDER BY with JOIN failed: {e}")
        
        # Test 4: Check if the issue is with specific columns
        print("\n🔧 Test 4: ORDER BY with specific columns")
        with db_manager.connection.cursor() as cursor:
            try:
                cursor.execute("""
                    SELECT pd.id, pd.file_name, pd.folder_name, pd.page_count,
                           pe.embedding_model,
                           pe.embedding_vector <=> %s::vector as distance
                    FROM pdf_documents pd
                    JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                    WHERE pe.embedding_vector IS NOT NULL
                    ORDER BY pe.embedding_vector <=> %s::vector
                    LIMIT 3
                """, (test_vector, test_vector))
                
                results = cursor.fetchall()
                print(f"   ✅ ORDER BY with specific columns returned {len(results)} results")
                for result in results:
                    print(f"     ID {result[0]}: {result[1]}, folder: {result[2]}, pages: {result[3]}, model: {result[4]}, distance: {result[5]:.6f}")
            except Exception as e:
                print(f"   ❌ ORDER BY with specific columns failed: {e}")
        
        # Test 5: Check if the issue is with the vector casting
        print("\n🔧 Test 5: Check vector casting")
        with db_manager.connection.cursor() as cursor:
            try:
                # First check what type the test_vector is being cast to
                cursor.execute("SELECT pg_typeof(%s::vector) as vector_type", (test_vector,))
                result = cursor.fetchone()
                print(f"   Test vector type: {result[0]}")
                
                # Check what type the stored vectors are
                cursor.execute("""
                    SELECT pg_typeof(pe.embedding_vector) as stored_vector_type
                    FROM pdf_embeddings pe
                    LIMIT 1
                """)
                result = cursor.fetchone()
                print(f"   Stored vector type: {result[0]}")
                
            except Exception as e:
                print(f"   ❌ Vector type check failed: {e}")
        
        print("\n✅ ORDER BY debugging complete!")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_order_by()
