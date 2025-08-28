#!/usr/bin/env python3
"""
Debug the JOIN condition in the PDF search query.
"""

import sys
from config import Config
from database_manager import DatabaseManager

def debug_join():
    """Debug the JOIN condition."""
    print("🔍 Debugging JOIN Condition")
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
        
        # Check PDF documents
        print("\n📋 PDF Documents:")
        with db_manager.connection.cursor() as cursor:
            cursor.execute("SELECT id, file_name FROM pdf_documents ORDER BY id")
            docs = cursor.fetchall()
            for doc in docs:
                print(f"   ID {doc[0]}: {doc[1]}")
        
        # Check PDF embeddings
        print("\n📋 PDF Embeddings:")
        with db_manager.connection.cursor() as cursor:
            cursor.execute("SELECT id, pdf_id, embedding_model FROM pdf_embeddings ORDER BY id")
            embs = cursor.fetchall()
            for emb in embs:
                print(f"   ID {emb[0]}, PDF ID {emb[1]}: {emb[2]}")
        
        # Check if there's a mismatch
        print("\n🔍 Checking for mismatches:")
        with db_manager.connection.cursor() as cursor:
            cursor.execute("""
                SELECT pd.id as doc_id, pe.pdf_id as emb_pdf_id
                FROM pdf_documents pd
                FULL OUTER JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                ORDER BY pd.id, pe.pdf_id
            """)
            matches = cursor.fetchall()
            for match in matches:
                doc_id = match[0]
                emb_pdf_id = match[1]
                if doc_id is None:
                    print(f"   ⚠️  Embedding {emb_pdf_id} has no matching document")
                elif emb_pdf_id is None:
                    print(f"   ⚠️  Document {doc_id} has no embedding")
                else:
                    print(f"   ✅ Document {doc_id} matches embedding {emb_pdf_id}")
        
        # Test the exact JOIN condition
        print("\n🔍 Testing exact JOIN condition:")
        with db_manager.connection.cursor() as cursor:
            cursor.execute("""
                SELECT pd.id, pe.pdf_id
                FROM pdf_documents pd
                JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                ORDER BY pd.id
            """)
            joins = cursor.fetchall()
            print(f"   ✅ JOIN returned {len(joins)} results")
            for join in joins:
                print(f"     Document {join[0]} -> Embedding {join[1]}")
        
        # Test the WHERE condition
        print("\n🔍 Testing WHERE condition:")
        with db_manager.connection.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as total_embeddings
                FROM pdf_embeddings
                WHERE embedding_vector IS NOT NULL
            """)
            result = cursor.fetchone()
            print(f"   Embeddings with non-null vectors: {result[0]}")
        
        # Test the complete query step by step
        print("\n🔍 Testing complete query step by step:")
        with db_manager.connection.cursor() as cursor:
            # Step 1: Just the JOIN
            cursor.execute("""
                SELECT COUNT(*) as join_count
                FROM pdf_documents pd
                JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
            """)
            result = cursor.fetchone()
            print(f"   Step 1 (JOIN): {result[0]} results")
            
            # Step 2: JOIN + WHERE
            cursor.execute("""
                SELECT COUNT(*) as where_count
                FROM pdf_documents pd
                JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                WHERE pe.embedding_vector IS NOT NULL
            """)
            result = cursor.fetchone()
            print(f"   Step 2 (JOIN + WHERE): {result[0]} results")
            
            # Step 3: Add similarity calculation
            test_vector = [0.1] * 768
            cursor.execute("""
                SELECT COUNT(*) as similarity_count
                FROM pdf_documents pd
                JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                WHERE pe.embedding_vector IS NOT NULL
                AND pe.embedding_vector <=> %s::vector IS NOT NULL
            """, (test_vector,))
            result = cursor.fetchone()
            print(f"   Step 3 (JOIN + WHERE + similarity): {result[0]} results")
        
        print("\n✅ JOIN debugging complete!")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_join()
