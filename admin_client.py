#!/usr/bin/env python3
"""
Admin Client for Service Administrators

Demonstrates how to use the admin API endpoints to manage entities.
"""

import requests
import json
from typing import Dict, Any, List, Optional


class AdminClient:
    """Client for interacting with the Admin API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()
    
    def login(self, username: str, password: str) -> bool:
        """Login to get authentication token."""
        try:
            response = self.session.post(f"{self.base_url}/auth/login", json={
                'username': username,
                'password': password
            })
            
            if response.status_code == 200:
                data = response.json()
                self.token = data['token']
                self.session.headers.update({'Authorization': f'Bearer {self.token}'})
                print(f"✅ Login successful for user: {data['user']['username']}")
                return True
            else:
                print(f"❌ Login failed: {response.json().get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def list_entities(self, entity_type: str) -> List[Dict[str, Any]]:
        """List all entities of a specific type."""
        try:
            response = self.session.get(f"{self.base_url}/admin/entities/{entity_type}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Found {data['count']} {entity_type}")
                return data['entities']
            else:
                print(f"❌ Failed to list {entity_type}: {response.json().get('error', 'Unknown error')}")
                return []
        
        except Exception as e:
            print(f"❌ Error listing {entity_type}: {e}")
            return []
    
    def get_entity(self, entity_type: str, entity_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific entity by ID."""
        try:
            response = self.session.get(f"{self.base_url}/admin/entities/{entity_type}/{entity_id}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Retrieved {entity_type} {entity_id}")
                return data['entity']
            else:
                print(f"❌ Failed to get {entity_type} {entity_id}: {response.json().get('error', 'Unknown error')}")
                return None
        
        except Exception as e:
            print(f"❌ Error getting {entity_type} {entity_id}: {e}")
            return None
    
    def update_entity(self, entity_type: str, entity_id: int, update_data: Dict[str, Any]) -> bool:
        """Update a specific entity."""
        try:
            response = self.session.put(
                f"{self.base_url}/admin/entities/{entity_type}/{entity_id}",
                json=update_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {entity_type.capitalize()} {entity_id} updated successfully")
                return True
            else:
                print(f"❌ Failed to update {entity_type} {entity_id}: {response.json().get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            print(f"❌ Error updating {entity_type} {entity_id}: {e}")
            return False
    
    def delete_entity(self, entity_type: str, entity_id: int) -> bool:
        """Delete a specific entity."""
        try:
            response = self.session.delete(f"{self.base_url}/admin/entities/{entity_type}/{entity_id}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {entity_type.capitalize()} {entity_id} deleted successfully")
                return True
            else:
                print(f"❌ Failed to delete {entity_type} {entity_id}: {response.json().get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            print(f"❌ Error deleting {entity_type} {entity_id}: {e}")
            return False
    
    def bulk_update_entities(self, entity_type: str, updates: List[Dict[str, Any]]) -> bool:
        """Bulk update multiple entities."""
        try:
            response = self.session.post(
                f"{self.base_url}/admin/bulk-update/{entity_type}",
                json={'updates': updates}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Bulk update completed for {data['total_updates']} {entity_type}")
                
                # Show results
                for result in data['results']:
                    if result['success']:
                        print(f"   ✅ {entity_type} {result['id']}: Updated successfully")
                    else:
                        print(f"   ❌ {entity_type} {result['id']}: {result['error']}")
                
                return True
            else:
                print(f"❌ Failed to bulk update {entity_type}: {response.json().get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            print(f"❌ Error bulk updating {entity_type}: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
        try:
            response = self.session.get(f"{self.base_url}/admin/system/status")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ System status retrieved successfully")
                return data
            else:
                print(f"❌ Failed to get system status: {response.json().get('error', 'Unknown error')}")
                return {}
        
        except Exception as e:
            print(f"❌ Error getting system status: {e}")
            return {}
    
    def perform_maintenance(self, action: str) -> bool:
        """Perform system maintenance tasks."""
        try:
            response = self.session.post(
                f"{self.base_url}/admin/system/maintenance",
                json={'action': action}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Maintenance action '{action}' completed: {data['message']}")
                return data['success']
            else:
                print(f"❌ Failed to perform maintenance action '{action}': {response.json().get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            print(f"❌ Error performing maintenance action '{action}': {e}")
            return False


def demo_admin_operations():
    """Demonstrate admin operations."""
    print("🚀 Admin API Client Demo")
    print("=" * 50)
    
    # Initialize client
    client = AdminClient()
    
    # Login
    print("\n1. 🔐 Admin Login")
    if not client.login('admin', 'admin123'):
        print("❌ Cannot proceed without authentication")
        return
    
    # Get system status
    print("\n2. 📊 System Status")
    status = client.get_system_status()
    if status:
        print(f"   Database connected: {status['database']['connected']}")
        print(f"   Total emails: {status['entities']['emails']}")
        print(f"   Total PDFs: {status['entities']['pdfs']}")
    
    # List entities
    print("\n3. 📧 List Emails")
    emails = client.list_entities('emails')
    if emails:
        print(f"   Found {len(emails)} emails")
        if emails:
            first_email = emails[0]
            print(f"   First email: ID {first_email['id']}, Subject: {first_email.get('subject', 'No subject')[:50]}...")
    
    # List PDFs
    print("\n4. 📄 List PDF Documents")
    pdfs = client.list_entities('pdfs')
    if pdfs:
        print(f"   Found {len(pdfs)} PDF documents")
        if pdfs:
            first_pdf = pdfs[0]
            print(f"   First PDF: ID {first_pdf['id']}, Name: {first_pdf.get('file_name', 'No name')[:50]}...")
    
    # Example update operation (commented out for safety)
    print("\n5. ✏️  Example Update Operation (commented out for safety)")
    print("   # Uncomment the following lines to test updates:")
    print("   # if emails:")
    print("   #     email_id = emails[0]['id']")
    print("   #     client.update_entity('emails', email_id, {'subject': 'Updated Subject'})")
    
    # System maintenance
    print("\n6. 🔧 System Maintenance")
    print("   Available actions: recompute_embeddings, cleanup_orphaned, optimize_database")
    print("   # Uncomment to test:")
    print("   # client.perform_maintenance('cleanup_orphaned')")
    
    print("\n✅ Demo completed successfully!")
    print("💡 Use the AdminClient class in your own scripts for automated administration")


if __name__ == "__main__":
    demo_admin_operations()

