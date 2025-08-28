#!/usr/bin/env python3
"""
Debug script for PDF search to identify why no results are returned.
"""

import sys
from config import Config
from database_manager import DatabaseManager
from database_searcher import DatabaseSearcher
from reranker import create_reranker

def debug_pdf_search():
    """Debug PDF search functionality."""
    print("🔍 Debugging PDF Search")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = Config()
        db_config = config.get_db_config()
        
        if not all(db_config.get(key) for key in ['host', 'database', 'user', 'password']):
            print("❌ Incomplete database configuration")
            return
        
        print(f"✅ Database config loaded: {db_config['host']}:{db_config['port']}/{db_config['database']}")
        
        # Initialize database manager
        db_manager = DatabaseManager(db_config)
        if not db_manager.connect():
            print("❌ Failed to connect to database")
            return
        
        print("✅ Database connection established")
        
        # Check if pgvector is available
        print(f"📊 pgvector available: {db_manager.pgvector_available}")
        
        # Check PDF documents table
        print("\n📋 Checking PDF documents table...")
        with db_manager.connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM pdf_documents")
            pdf_count = cursor.fetchone()[0]
            print(f"   Total PDF documents: {pdf_count}")
            
            if pdf_count > 0:
                cursor.execute("SELECT id, file_name, folder_name FROM pdf_documents LIMIT 5")
                pdfs = cursor.fetchall()
                print("   Sample PDFs:")
                for pdf in pdfs:
                    print(f"     ID {pdf[0]}: {pdf[1]} (folder: {pdf[2]})")
        
        # Check PDF embeddings table
        print("\n🔢 Checking PDF embeddings table...")
        with db_manager.connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM pdf_embeddings")
            embedding_count = cursor.fetchone()[0]
            print(f"   Total PDF embeddings: {embedding_count}")
            
            if embedding_count > 0:
                cursor.execute("SELECT pdf_id, embedding_model FROM pdf_embeddings LIMIT 5")
                embeddings = cursor.fetchall()
                print("   Sample embeddings:")
                for emb in embeddings:
                    print(f"     PDF ID {emb[0]}: {emb[1]}")
        
        # Check for PDFs without embeddings
        print("\n🔍 Checking PDFs without embeddings...")
        pdfs_without_embeddings = db_manager.get_pdf_documents_without_embeddings(limit=5)
        print(f"   PDFs without embeddings: {len(pdfs_without_embeddings)}")
        if pdfs_without_embeddings:
            print("   Sample PDFs without embeddings:")
            for pdf in pdfs_without_embeddings[:3]:
                print(f"     ID {pdf['id']}: {pdf['file_name']}")
        
        # Test direct database query
        print("\n🧪 Testing direct database query...")
        with db_manager.connection.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM pdf_documents pd
                JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                WHERE pe.embedding_vector IS NOT NULL
            """)
            count = cursor.fetchone()[0]
            print(f"   PDFs with valid embeddings: {count}")
            
            if count > 0:
                cursor.execute("""
                    SELECT pd.id, pd.file_name, pe.embedding_model
                    FROM pdf_documents pd
                    JOIN pdf_embeddings pe ON pd.id = pe.pdf_id
                    WHERE pe.embedding_vector IS NOT NULL
                    LIMIT 3
                """)
                results = cursor.fetchall()
                print("   Sample PDFs with embeddings:")
                for result in results:
                    print(f"     ID {result[0]}: {result[1]} ({result[2]})")
        
        # Test semantic search if we have embeddings
        if embedding_count > 0:
            print("\n🔍 Testing semantic search...")
            
            # Initialize searcher
            embedding_model = config.get_embedding_model()
            reranker = create_reranker()
            searcher = DatabaseSearcher(db_manager, embedding_model, reranker)
            
            # Test search
            test_query = "test query"
            print(f"   Testing search with query: '{test_query}'")
            
            results = searcher.search_pdf_documents(
                query=test_query,
                limit=5
            )
            
            print(f"   Search returned {len(results)} results")
            if results:
                print("   First result:")
                first = results[0]
                print(f"     ID: {first.get('id')}")
                print(f"     File: {first.get('file_name')}")
                print(f"     Similarity: {first.get('similarity_score', 'N/A')}")
        
        print("\n✅ Debug complete!")
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pdf_search()
