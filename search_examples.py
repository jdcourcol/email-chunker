#!/usr/bin/env python3
"""
Search Examples for Enhanced Maildir Parser

This script demonstrates the new hybrid search functionality combining
SQL LIKE search and semantic search with embeddings.
"""

from enhanced_parser import EnhancedMaildirParser, DatabaseManager
from database_manager import create_embedding_model
from config import Config


def example_hybrid_search(maildir_path: str = None):
    """Example of hybrid search functionality."""
    print("=== Hybrid Search Examples ===")
    
    try:
        # Load configuration
        config = Config()
        if not config.has_db_config():
            print("Error: No database configuration found")
            print("Please run: python config.py setup")
            return
        
        # Initialize database manager
        db_manager = DatabaseManager(config.get_db_config())
        if not db_manager.create_tables():
            print("Failed to create database tables")
            return
        
        # Initialize embedding model
        embedding_model = create_embedding_model()
        
        # Initialize enhanced parser
        parser = EnhancedMaildirParser(
            maildir_path,
            convert_html=True,
            aggressive_clean=True,
            db_manager=db_manager,
            embedding_model=embedding_model
        )
        
        # Example 1: SQL-only search
        print("\n1. SQL-only search for 'meeting':")
        results = parser.search_emails_hybrid(
            query="meeting",
            search_type="sql",
            search_fields=["subject", "body"],
            limit=5
        )
        display_search_results(results)
        
        # Example 2: Semantic-only search
        print("\n2. Semantic-only search for 'project deadline':")
        results = parser.search_emails_hybrid(
            query="project deadline",
            search_type="semantic",
            limit=5
        )
        display_search_results(results)
        
        # Example 3: Combined search (both SQL and semantic)
        print("\n3. Combined search for 'urgent':")
        results = parser.search_emails_hybrid(
            query="urgent",
            search_type="both",
            search_fields=["subject", "body", "sender"],
            limit=10
        )
        display_search_results(results)
        
        # Example 4: Search in specific folder
        print("\n4. Search in INBOX folder only:")
        results = parser.search_emails_hybrid(
            query="meeting",
            search_type="both",
            folder="INBOX",
            limit=5
        )
        display_search_results(results)
        
        # Example 5: Custom search fields
        print("\n5. Search only in subject and sender fields:")
        results = parser.search_emails_hybrid(
            query="john",
            search_type="sql",
            search_fields=["subject", "sender"],
            limit=5
        )
        display_search_results(results)
        
        # Clean up
        if maildir_path:
            parser.close_mailbox()
        db_manager.disconnect()
        
    except Exception as e:
        print(f"Error: {e}")


def example_search_comparison(maildir_path: str = None):
    """Compare different search types for the same query."""
    print("\n=== Search Type Comparison ===")
    
    try:
        # Load configuration
        config = Config()
        if not config.has_db_config():
            print("Error: No database configuration found")
            print("Please run: python config.py setup")
            return
        
        # Initialize components
        db_manager = DatabaseManager(config.get_db_config())
        embedding_model = create_embedding_model()
        parser = EnhancedMaildirParser(
            maildir_path,
            db_manager=db_manager,
            embedding_model=embedding_model
        )
        
        query = "meeting schedule"
        print(f"Comparing search types for query: '{query}'")
        
        # SQL search
        print("\n1. SQL LIKE Search:")
        sql_results = parser.search_emails_hybrid(
            query=query,
            search_type="sql",
            search_fields=["subject", "body"],
            limit=5
        )
        display_search_results(sql_results)
        
        # Semantic search
        print("\n2. Semantic Search:")
        semantic_results = parser.search_emails_hybrid(
            query=query,
            search_type="semantic",
            limit=5
        )
        display_search_results(semantic_results)
        
        # Combined search
        print("\n3. Combined Search:")
        combined_results = parser.search_emails_hybrid(
            query=query,
            search_type="both",
            search_fields=["subject", "body", "sender"],
            limit=10
        )
        display_search_results(combined_results)
        
        # Clean up
        if maildir_path:
            parser.close_mailbox()
        db_manager.disconnect()
        
    except Exception as e:
        print(f"Error: {e}")


def display_search_results(results: dict):
    """Display search results in a formatted way."""
    if not results:
        print("  No results found")
        return
    
    metadata = results['search_metadata']
    print(f"  Search Type: {metadata['search_type']}")
    print(f"  Folder: {metadata.get('folder', 'All folders')}")
    print(f"  SQL Results: {metadata.get('sql_count', 0)}")
    print(f"  Semantic Results: {metadata.get('semantic_count', 0)}")
    print(f"  Total Results: {metadata['total_results']}")
    
    # Show results based on search type
    if metadata['search_type'] == 'both' and results['combined_results']:
        emails_to_show = results['combined_results']
        print(f"  Combined Results (Top {len(emails_to_show)}):")
    elif metadata['search_type'] == 'sql' and results['sql_results']:
        emails_to_show = results['sql_results']
        print(f"  SQL Results (Top {len(emails_to_show)}):")
    elif metadata['search_type'] == 'semantic' and results['semantic_results']:
        emails_to_show = results['semantic_results']
        print(f"  Semantic Results (Top {len(emails_to_show)}):")
    else:
        emails_to_show = []
        print("  No results to display")
        return
    
    for i, email in enumerate(emails_to_show[:3], 1):  # Show top 3
        search_type = email.get('search_type', 'unknown')
        similarity = email.get('similarity_score', 0)
        
        print(f"    {i}. {email.get('subject', 'No subject')[:50]}...")
        print(f"       From: {email.get('sender', 'Unknown')}")
        print(f"       Folder: {email.get('folder', 'Unknown')}")
        if search_type == 'semantic' and similarity > 0:
            print(f"       Similarity: {similarity:.3f}")
        print(f"       Search Type: {search_type}")
    
    if len(emails_to_show) > 3:
        print(f"    ... and {len(emails_to_show) - 3} more results")


def main():
    """Main function to run search examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Search Examples for Enhanced Maildir Parser')
    parser.add_argument('maildir_path', nargs='?', help='Path to the Maildir root folder (optional)')
    
    # Example options
    parser.add_argument('--hybrid-search', action='store_true', help='Run hybrid search examples')
    parser.add_argument('--search-comparison', action='store_true', help='Compare search types')
    parser.add_argument('--run-all', action='store_true', help='Run all examples')
    
    args = parser.parse_args()
    
    try:
        if args.hybrid_search or args.run_all:
            example_hybrid_search(args.maildir_path)
        
        if args.search_comparison or args.run_all:
            example_search_comparison(args.maildir_path)
        
        if not any([args.hybrid_search, args.search_comparison, args.run_all]):
            print("No examples selected. Use --hybrid-search, --search-comparison, or --run-all")
            print("\nAvailable examples:")
            print("  --hybrid-search: Demonstrate hybrid search functionality")
            print("  --search-comparison: Compare different search types")
            print("  --run-all: Run all examples")
            print("\nNote: Database configuration is loaded automatically from config.json")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
