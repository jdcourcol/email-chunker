#!/usr/bin/env python3
"""
Maildir Email Parser

This script reads and parses emails from a Maildir folder structure.
Maildir is a standard format for storing emails in a filesystem hierarchy.
Supports multiple subdirectories like INBOX, Drafts, Sent, etc.
Includes PostgreSQL database integration and sentence embeddings.
"""

import mailbox
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import re
import html
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import numpy as np


def html_to_plain_text(html_content: str) -> str:
    """
    Convert HTML content to plain text.
    
    Args:
        html_content: HTML string to convert
        
    Returns:
        Plain text version of the HTML content
    """
    if not html_content:
        return ""
    
    # Decode HTML entities
    text = html.unescape(html_content)
    
    # Remove CSS style blocks and script blocks
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove HTML tags with their attributes
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove CSS properties and values that might leak through
    text = re.sub(r'[a-zA-Z\-]+\s*:\s*[^;]+;?', '', text)
    
    # Remove CSS class names and IDs that might be left over
    text = re.sub(r'\.\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove CSS at-rules
    text = re.sub(r'@[^{]+{[^}]*}', '', text)
    
    # Remove any remaining CSS-like patterns
    text = re.sub(r'[a-zA-Z\-]+\s*{[^}]*}', '', text)
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Clean up common HTML artifacts
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    
    # Remove any remaining CSS selectors or properties
    text = re.sub(r'[a-zA-Z\-]+\s*:\s*[^;\s]+', '', text)
    
    # Remove any remaining HTML-like fragments
    text = re.sub(r'[<>]', '', text)
    
    # Clean up multiple spaces and normalize
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


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


class MaildirParser:
    """A class to parse emails from a Maildir folder structure."""
    
    def __init__(self, maildir_path: str, convert_html: bool = True, aggressive_clean: bool = False, 
                 db_manager: Optional[DatabaseManager] = None, embedding_model=None):
        """
        Initialize the MaildirParser.
        
        Args:
            maildir_path: Path to the Maildir folder (root directory containing subdirectories)
            convert_html: Whether to convert HTML content to plain text (default: True)
            aggressive_clean: Whether to use aggressive CSS/HTML cleaning (default: False)
            db_manager: Optional database manager for saving emails
            embedding_model: Optional sentence transformer model for embeddings
        """
        self.maildir_path = Path(maildir_path)
        self.mailbox = None
        self.current_folder = None
        self.convert_html = convert_html
        self.aggressive_clean = aggressive_clean
        self.db_manager = db_manager
        self.embedding_model = embedding_model
    
    def list_folders(self) -> List[str]:
        """
        List all available Maildir folders (subdirectories).
        
        Returns:
            List of folder names
        """
        folders = []
        for item in self.maildir_path.iterdir():
            if item.is_dir():
                # Check if it's a valid Maildir folder (has new/ and cur/ subdirectories)
                if (item / 'new').exists() and (item / 'cur').exists():
                    folders.append(item.name)
        return sorted(folders)
    
    def open_folder(self, folder_name: str) -> bool:
        """
        Open a specific Maildir folder.
        
        Args:
            folder_name: Name of the folder to open (e.g., 'INBOX', 'Drafts')
            
        Returns:
            True if successful, False otherwise
        """
        folder_path = self.maildir_path / folder_name
        
        if not folder_path.exists():
            print(f"Folder '{folder_name}' does not exist")
            return False
        
        if not folder_path.is_dir():
            print(f"'{folder_name}' is not a directory")
            return False
        
        # Check if it has the required Maildir structure
        if not (folder_path / 'new').exists() or not (folder_path / 'cur').exists():
            print(f"'{folder_name}' is not a valid Maildir folder")
            return False
        
        try:
            if self.mailbox:
                self.mailbox.close()
            
            self.mailbox = mailbox.Maildir(folder_path, factory=None)
            self.current_folder = folder_name
            return True
        except Exception as e:
            print(f"Error opening folder '{folder_name}': {e}")
            return False
    
    def open_mailbox(self):
        """Open the default mailbox (INBOX if available, otherwise first folder)."""
        folders = self.list_folders()
        
        if not folders:
            print("No valid Maildir folders found")
            return False
        
        # Try to open INBOX first, then fall back to first available folder
        if 'INBOX' in folders:
            return self.open_folder('INBOX')
        else:
            return self.open_folder(folders[0])
    
    def close_mailbox(self):
        """Close the mailbox."""
        if self.mailbox:
            self.mailbox.close()
            self.mailbox = None
            self.current_folder = None
    
    def get_current_folder(self) -> Optional[str]:
        """Get the name of the currently open folder."""
        return self.current_folder
    
    def get_email_count(self) -> int:
        """Get the total number of emails in the currently open folder."""
        if not self.mailbox:
            if not self.open_mailbox():
                return 0
        return len(self.mailbox)
    
    def get_folder_email_count(self, folder_name: str) -> int:
        """
        Get the email count for a specific folder without opening it.
        
        Args:
            folder_name: Name of the folder
            
        Returns:
            Number of emails in the folder
        """
        folder_path = self.maildir_path / folder_name
        if not folder_path.exists():
            return 0
        
        try:
            temp_mailbox = mailbox.Maildir(folder_path, factory=None)
            count = len(temp_mailbox)
            temp_mailbox.close()
            return count
        except Exception:
            return 0
    
    def get_all_folders_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all folders including email counts.
        
        Returns:
            Dictionary with folder information
        """
        folders_info = {}
        for folder_name in self.list_folders():
            count = self.get_folder_email_count(folder_name)
            folders_info[folder_name] = {
                'name': folder_name,
                'email_count': count,
                'path': str(self.maildir_path / folder_name)
            }
        return folders_info
    
    def parse_email(self, message) -> Dict[str, Any]:
        """
        Parse a single email message.
        
        Args:
            message: Email message object
            
        Returns:
            Dictionary containing parsed email data
        """
        parsed_email = {
            'subject': '',
            'from': '',
            'to': '',
            'date': '',
            'message_id': '',
            'content_type': '',
            'body': '',
            'attachments': [],
            'headers': {},
            'folder': self.current_folder
        }
        
        try:
            # Parse headers
            parsed_email['subject'] = message.get('subject', '')
            parsed_email['from'] = message.get('from', '')
            parsed_email['to'] = message.get('to', '')
            parsed_email['date'] = message.get('date', '')
            parsed_email['message_id'] = message.get('message-id', '')
            
            # Get all headers
            for header_name in message.keys():
                parsed_email['headers'][header_name] = message.get(header_name, '')
            
            # Parse body and attachments
            if message.is_multipart():
                parsed_email['content_type'] = 'multipart'
                for part in message.walk():
                    if part.is_multipart():
                        continue
                    
                    content_type = part.get_content_type()
                    if content_type == 'text/plain':
                        try:
                            parsed_email['body'] = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        except:
                            parsed_email['body'] = str(part.get_payload())
                    elif content_type == 'text/html':
                        if not parsed_email['body']:  # Prefer plain text over HTML
                            try:
                                html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                if self.aggressive_clean:
                                    parsed_email['body'] = html_to_plain_text(html_content)
                                    # Additional aggressive cleaning for CSS artifacts
                                    body = parsed_email['body']
                                    # Remove any remaining CSS-like patterns
                                    body = re.sub(r'[a-zA-Z\-]+\s*{[^}]*}', '', body)
                                    body = re.sub(r'[a-zA-Z\-]+\s*:\s*[^;\s]+', '', body)
                                    body = re.sub(r'\.\w+', '', body)
                                    body = re.sub(r'#\w+', '', body)
                                    body = re.sub(r'\s+', ' ', body)
                                    parsed_email['body'] = body.strip()
                                    parsed_email['content_type'] = 'text/plain (converted from HTML, aggressively cleaned)'
                                else:
                                    parsed_email['body'] = html_to_plain_text(html_content)
                                    parsed_email['content_type'] = 'text/plain (converted from HTML)'
                            except:
                                parsed_email['body'] = str(part.get_payload())
                    else:
                        # This is an attachment
                        filename = part.get_filename()
                        if filename:
                            try:
                                content = part.get_payload(decode=True)
                                size = len(content) if content else 0
                            except:
                                size = 0
                            parsed_email['attachments'].append({
                                'filename': filename,
                                'content_type': content_type,
                                'size': size
                            })
            else:
                # Single part message
                content_type = message.get_content_type()
                parsed_email['content_type'] = content_type
                
                try:
                    content = message.get_payload(decode=True).decode('utf-8', errors='ignore')
                    if content_type == 'text/html' and self.convert_html:
                        # Convert HTML to plain text
                        if self.aggressive_clean:
                            parsed_email['body'] = html_to_plain_text(content)
                            # Additional aggressive cleaning for CSS artifacts
                            body = parsed_email['body']
                            # Remove any remaining CSS-like patterns
                            body = re.sub(r'[a-zA-Z\-]+\s*{[^}]*}', '', body)
                            body = re.sub(r'[a-zA-Z\-]+\s*:\s*[^;\s]+', '', body)
                            body = re.sub(r'\.\w+', '', body)
                            body = re.sub(r'#\w+', '', body)
                            body = re.sub(r'\s+', ' ', body)
                            parsed_email['body'] = body.strip()
                            parsed_email['content_type'] = 'text/plain (converted from HTML, aggressively cleaned)'
                        else:
                            parsed_email['body'] = html_to_plain_text(content)
                            parsed_email['content_type'] = 'text/plain (converted from HTML)'
                    else:
                        parsed_email['body'] = content
                except:
                    parsed_email['body'] = str(message.get_payload())
            
            return parsed_email
            
        except Exception as e:
            print(f"Error parsing email: {e}")
            return parsed_email
    
    def get_all_emails(self) -> List[Dict[str, Any]]:
        """
        Get all emails from the currently open folder.
        
        Returns:
            List of parsed email dictionaries
        """
        if not self.mailbox:
            if not self.open_mailbox():
                return []
        
        emails = []
        for key, message in self.mailbox.items():
            try:
                parsed_email = self.parse_email(message)
                emails.append(parsed_email)
            except Exception as e:
                print(f"Error processing email {key}: {e}")
                continue
        
        return emails
    
    def get_emails_from_folder(self, folder_name: str) -> List[Dict[str, Any]]:
        """
        Get all emails from a specific folder.
        
        Args:
            folder_name: Name of the folder to read
            
        Returns:
            List of parsed email dictionaries
        """
        if self.open_folder(folder_name):
            return self.get_all_emails()
        return []
    
    def get_email_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific email by its key from the currently open folder.
        
        Args:
            key: Email key identifier
            
        Returns:
            Parsed email dictionary or None if not found
        """
        if not self.mailbox:
            if not self.open_mailbox():
                return None
        
        try:
            message = self.mailbox[key]
            return self.parse_email(message)
        except KeyError:
            print(f"Email with key '{key}' not found")
            return None
        except Exception as e:
            print(f"Error getting email {key}: {e}")
            return None
    
    def search_emails(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Search emails in the currently open folder by various criteria.
        
        Args:
            **kwargs: Search criteria (subject, from, to, etc.)
            
        Returns:
            List of matching emails
        """
        all_emails = self.get_all_emails()
        matching_emails = []
        
        for email_data in all_emails:
            match = True
            for key, value in kwargs.items():
                if key in email_data:
                    if isinstance(email_data[key], str) and isinstance(value, str):
                        if value.lower() not in email_data[key].lower():
                            match = False
                            break
                    elif email_data[key] != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                matching_emails.append(email_data)
        
        return matching_emails
    
    def search_all_folders(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Search emails across all folders by various criteria.
        
        Args:
            **kwargs: Search criteria (subject, from, to, etc.)
            
        Returns:
            List of matching emails from all folders
        """
        all_matching_emails = []
        
        for folder_name in self.list_folders():
            if self.open_folder(folder_name):
                matching_emails = self.search_emails(**kwargs)
                all_matching_emails.extend(matching_emails)
        
        return all_matching_emails
    
    def save_emails_to_database(self, emails: List[Dict[str, Any]], 
                               compute_embeddings: bool = True) -> Dict[str, int]:
        """
        Save emails to database and optionally compute embeddings.
        
        Args:
            emails: List of parsed email dictionaries
            compute_embeddings: Whether to compute and save embeddings
            
        Returns:
            Dictionary with save statistics
        """
        if not self.db_manager:
            print("No database manager configured")
            return {'saved': 0, 'skipped': 0, 'embeddings': 0}
        
        saved_count = 0
        skipped_count = 0
        embedding_count = 0
        
        for email_data in emails:
            message_id = email_data.get('message_id', '')
            if not message_id:
                print("Skipping email without message ID")
                skipped_count += 1
                continue
            
            # Check if email already exists
            if self.db_manager.email_exists(message_id):
                print(f"Email already exists: {message_id[:50]}...")
                skipped_count += 1
                continue
            
            # Save email to database
            email_id = self.db_manager.save_email(email_data)
            if email_id:
                saved_count += 1
                print(f"Saved email: {email_data.get('subject', 'No subject')[:50]}...")
                
                # Compute and save embedding if requested
                if compute_embeddings and self.embedding_model:
                    try:
                        # Create text for embedding (subject + body)
                        text_for_embedding = f"{email_data.get('subject', '')} {email_data.get('body', '')}"
                        text_for_embedding = text_for_embedding.strip()
                        
                        if text_for_embedding:
                            # Compute embedding
                            embedding = self.embedding_model.encode(text_for_embedding).tolist()
                            
                            # Save embedding to database
                            if self.db_manager.save_embedding(email_id, embedding, 'all-MiniLM-L6-v2'):
                                embedding_count += 1
                                print(f"  Saved embedding for email ID {email_id}")
                            else:
                                print(f"  Failed to save embedding for email ID {email_id}")
                    except Exception as e:
                        print(f"  Error computing embedding for email ID {email_id}: {e}")
            else:
                print(f"Failed to save email: {email_data.get('subject', 'No subject')[:50]}...")
                skipped_count += 1
        
        return {
            'saved': saved_count,
            'skipped': skipped_count,
            'embeddings': embedding_count
        }
    
    def process_folder_to_database(self, folder_name: str, compute_embeddings: bool = True) -> Dict[str, int]:
        """
        Process all emails from a specific folder and save to database.
        
        Args:
            folder_name: Name of the folder to process
            compute_embeddings: Whether to compute and save embeddings
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"Processing folder: {folder_name}")
        
        if not self.open_folder(folder_name):
            return {'saved': 0, 'skipped': 0, 'embeddings': 0}
        
        emails = self.get_all_emails()
        print(f"Found {len(emails)} emails in {folder_name}")
        
        return self.save_emails_to_database(emails, compute_embeddings)
    
    def process_all_folders_to_database(self, compute_embeddings: bool = True) -> Dict[str, int]:
        """
        Process all folders and save emails to database.
        
        Args:
            compute_embeddings: Whether to compute and save embeddings
            
        Returns:
            Dictionary with processing statistics
        """
        total_stats = {'saved': 0, 'skipped': 0, 'embeddings': 0}
        
        for folder_name in self.list_folders():
            folder_stats = self.process_folder_to_database(folder_name, compute_embeddings)
            
            total_stats['saved'] += folder_stats['saved']
            total_stats['skipped'] += folder_stats['skipped']
            total_stats['embeddings'] += folder_stats['embeddings']
            
            print(f"Folder {folder_name}: {folder_stats['saved']} saved, {folder_stats['skipped']} skipped")
        
        return total_stats


def print_email_summary(email_data: Dict[str, Any], index: int = None):
    """Print a summary of an email."""
    prefix = f"[{index}] " if index is not None else ""
    folder_info = f" (Folder: {email_data.get('folder', 'Unknown')})"
    print(f"{prefix}Subject: {email_data['subject']}")
    print(f"{' ' * len(prefix)}From: {email_data['from']}")
    print(f"{' ' * len(prefix)}Date: {email_data['date']}")
    print(f"{' ' * len(prefix)}Attachments: {len(email_data['attachments'])}")
    print(f"{' ' * len(prefix)}Folder: {email_data.get('folder', 'Unknown')}")
    if email_data['body']:
        body_preview = email_data['body'][:100].replace('\n', ' ')
        print(f"{' ' * len(prefix)}Body: {body_preview}...")
    print("-" * 50)


def print_folder_info(folders_info: Dict[str, Dict[str, Any]]):
    """Print information about all folders."""
    print("Available Maildir folders:")
    print("=" * 50)
    for folder_name, info in folders_info.items():
        print(f"{folder_name:15} - {info['email_count']:4d} emails")
    print("=" * 50)


def main():
    """Main function to demonstrate usage."""
    parser = argparse.ArgumentParser(description='Parse emails from a Maildir folder structure')
    parser.add_argument('maildir_path', help='Path to the Maildir root folder (containing INBOX, Drafts, etc.)')
    parser.add_argument('--folder', help='Specific folder to process (e.g., INBOX, Drafts)')
    parser.add_argument('--list-folders', action='store_true', help='List all available folders and email counts')
    parser.add_argument('--search', help='Search term for subject, from, or to fields')
    parser.add_argument('--search-all', action='store_true', help='Search across all folders (use with --search)')
    parser.add_argument('--show-body', action='store_true', help='Show full email body')
    parser.add_argument('--limit', type=int, help='Limit number of emails to display')
    parser.add_argument('--no-html-convert', action='store_true', help='Keep HTML content as-is (default: convert to plain text)')
    parser.add_argument('--aggressive-clean', action='store_true', help='Use aggressive CSS/HTML cleaning (removes more artifacts)')
    
    # Database options
    parser.add_argument('--db-host', help='PostgreSQL database host')
    parser.add_argument('--db-port', type=int, default=5432, help='PostgreSQL database port (default: 5432)')
    parser.add_argument('--db-name', help='PostgreSQL database name')
    parser.add_argument('--db-user', help='PostgreSQL database user')
    parser.add_argument('--db-password', help='PostgreSQL database password')
    
    # Processing options
    parser.add_argument('--save-to-db', action='store_true', help='Save emails to database')
    parser.add_argument('--process-all', action='store_true', help='Process all folders and save to database')
    parser.add_argument('--no-embeddings', action='store_true', help='Skip computing embeddings when saving to database')
    
    args = parser.parse_args()
    
    try:
        # Initialize database manager if database parameters provided
        db_manager = None
        if args.db_host and args.db_name and args.db_user:
            db_params = {
                'host': args.db_host,
                'port': args.db_port,
                'database': args.db_name,
                'user': args.db_user,
                'password': args.db_password
            }
            db_manager = DatabaseManager(db_params)
            
            if not db_manager.create_tables():
                print("Failed to create database tables")
                return 1
            
            print("Database connection established and tables created")
        
        # Initialize embedding model if saving to database
        embedding_model = None
        if args.save_to_db and not args.no_embeddings:
            try:
                print("Loading sentence transformer model...")
                from sentence_transformers import SentenceTransformer
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Sentence transformer model loaded successfully")
            except ImportError:
                print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
                print("Continuing without embeddings...")
            except Exception as e:
                print(f"Warning: Failed to load embedding model: {e}")
                print("Continuing without embeddings...")
        
        # Initialize parser
        convert_html = not args.no_html_convert
        aggressive_clean = args.aggressive_clean
        mail_parser = MaildirParser(
            args.maildir_path, 
            convert_html=convert_html, 
            aggressive_clean=aggressive_clean,
            db_manager=db_manager,
            embedding_model=embedding_model
        )
        
        # Process all folders to database if requested
        if args.process_all and db_manager:
            print("Processing all folders to database...")
            stats = mail_parser.process_all_folders_to_database(compute_embeddings=not args.no_embeddings)
            print(f"\nProcessing complete:")
            print(f"  Total emails saved: {stats['saved']}")
            print(f"  Total emails skipped: {stats['skipped']}")
            print(f"  Total embeddings created: {stats['embeddings']}")
            
            if db_manager:
                db_count = db_manager.get_email_count()
                print(f"  Total emails in database: {db_count}")
            
            return 0
        
        # List folders if requested
        if args.list_folders:
            folders_info = mail_parser.get_all_folders_info()
            print_folder_info(folders_info)
            return 0
        
        # Open specific folder or default
        if args.folder:
            if not mail_parser.open_folder(args.folder):
                print(f"Could not open folder '{args.folder}'")
                return 1
        else:
            if not mail_parser.open_mailbox():
                print("Could not open any mailbox")
                return 1
        
        current_folder = mail_parser.get_current_folder()
        print(f"Processing folder: {current_folder}")
        
        # Get email count
        email_count = mail_parser.get_email_count()
        print(f"Found {email_count} emails in {current_folder}")
        print("=" * 60)
        
        # Save to database if requested
        if args.save_to_db and db_manager:
            print("Saving emails to database...")
            emails = mail_parser.get_all_emails()
            stats = mail_parser.save_emails_to_database(emails, compute_embeddings=not args.no_embeddings)
            print(f"Database save complete:")
            print(f"  Emails saved: {stats['saved']}")
            print(f"  Emails skipped: {stats['skipped']}")
            print(f"  Embeddings created: {stats['embeddings']}")
            
            if db_manager:
                db_count = db_manager.get_email_count()
                print(f"  Total emails in database: {db_count}")
            
            return 0
        
        if args.search:
            # Search for emails
            print(f"Searching for emails containing: {args.search}")
            
            if args.search_all:
                # Search across all folders
                matching_emails = mail_parser.search_all_folders(subject=args.search)
                print(f"Searching across all folders...")
            else:
                # Search in current folder only
                matching_emails = mail_parser.search_emails(subject=args.search)
            
            # Remove duplicates if searching across folders
            if args.search_all:
                seen = set()
                unique_emails = []
                for email in matching_emails:
                    email_id = email.get('message_id', '')
                    if email_id not in seen:
                        seen.add(email_id)
                        unique_emails.append(email)
                matching_emails = unique_emails
            
            print(f"Found {len(matching_emails)} matching emails")
            emails_to_display = matching_emails
        else:
            # Get all emails from current folder
            emails_to_display = mail_parser.get_all_emails()
        
        # Apply limit if specified
        if args.limit:
            emails_to_display = emails_to_display[:args.limit]
        
        # Display emails
        for i, email_data in enumerate(emails_to_display):
            print_email_summary(email_data, i)
            
            if args.show_body and email_data['body']:
                print("Full Body:")
                print(email_data['body'])
                print("=" * 60)
        
        # Clean up
        mail_parser.close_mailbox()
        if db_manager:
            db_manager.disconnect()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
