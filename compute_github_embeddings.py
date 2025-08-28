#!/usr/bin/env python3
"""
Compute embeddings for GitHub issues

This script computes embeddings for GitHub issues that are already stored in the database
but don't have embeddings yet.
"""

import sys
from typing import List, Dict, Any
from config import Config
from database_manager import DatabaseManager
from embedding_model import create_embedding_model

def compute_github_issue_embeddings():
    """Compute embeddings for GitHub issues."""
    print("🚀 Computing GitHub Issue Embeddings")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config = Config()
        db_config = config.get_db_config()
        
        if not all(db_config.get(key) for key in ['host', 'database', 'user', 'password']:
            print("❌ Incomplete database configuration")
            return False
        
        print(f"✅ Database config loaded")
        
        # Initialize database manager
        db_manager = DatabaseManager(db_config)
        if not db_manager.connect():
            print("❌ Failed to connect to database")
            return False
        
        print("✅ Database connection established")
        
        # Load embedding model
        embedding_model = create_embedding_model()
        if not embedding_model:
            print("❌ Failed to load embedding model")
            return False
        
        print(f"✅ Embedding model loaded: {config.get_embedding_model()}")
        
        # Get GitHub issues without embeddings
        issues_without_embeddings = get_issues_without_embeddings(db_manager)
        
        if not issues_without_embeddings:
            print("✅ All GitHub issues already have embeddings!")
            return True
        
        print(f"📊 Found {len(issues_without_embeddings)} issues without embeddings")
        print()
        
        # Compute embeddings
        success_count = 0
        for i, issue in enumerate(issues_without_embeddings, 1):
            print(f"🔧 Processing issue {i}/{len(issues_without_embeddings)}: {issue['title'][:60]}...")
            
            # Create text for embedding (title + body)
            text_for_embedding = f"{issue['title']}\n\n{issue['body']}"
            
            try:
                # Compute embedding
                embedding = embedding_model.encode(text_for_embedding).tolist()
                
                # Store embedding
                if db_manager.store_github_issue_embedding(
                    issue['issue_id'], 
                    config.get_embedding_model(), 
                    embedding
                ):
                    success_count += 1
                    print(f"   ✅ Embedding computed and stored")
                else:
                    print(f"   ❌ Failed to store embedding")
                    
            except Exception as e:
                print(f"   ❌ Error computing embedding: {e}")
            
            print()
        
        print("=" * 50)
        print(f"✅ Completed! Successfully processed {success_count}/{len(issues_without_embeddings)} issues")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_issues_without_embeddings(db_manager: DatabaseManager) -> List[Dict[str, Any]]:
    """Get GitHub issues that don't have embeddings yet."""
    try:
        with db_manager.connection.cursor(cursor_factory=db_manager.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT gi.issue_id, gi.title, gi.body
                FROM github_issues gi
                LEFT JOIN github_issue_embeddings gie ON gi.issue_id = gie.issue_id
                WHERE gie.issue_id IS NULL
                ORDER BY gi.updated_at DESC
            """)
            return cursor.fetchall()
    except Exception as e:
        print(f"Error getting issues without embeddings: {e}")
        return []

def main():
    """Main function."""
    success = compute_github_issue_embeddings()
    
    if success:
        print("🎉 GitHub issue embeddings computed successfully!")
        return 0
    else:
        print("❌ Failed to compute GitHub issue embeddings")
        return 1

if __name__ == "__main__":
    exit(main())
