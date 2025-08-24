#!/usr/bin/env python3
"""
Example usage of the MaildirParser class.

This script demonstrates how to use the MaildirParser to read and process
emails from a Maildir folder structure with multiple subdirectories.
"""

from main import MaildirParser
import json


def example_list_folders(maildir_path: str):
    """Example of listing all available Maildir folders."""
    print("=== List Folders Example ===")
    
    try:
        parser = MaildirParser(maildir_path)
        
        # List all folders with email counts
        folders_info = parser.get_all_folders_info()
        
        print(f"Found {len(folders_info)} Maildir folders:")
        for folder_name, info in folders_info.items():
            print(f"  {folder_name:15} - {info['email_count']:4d} emails")
        
        # Clean up
        parser.close_mailbox()
        
    except Exception as e:
        print(f"Error: {e}")


def example_process_specific_folder(maildir_path: str, folder_name: str):
    """Example of processing a specific folder."""
    print(f"=== Process Specific Folder Example: {folder_name} ===")
    
    try:
        parser = MaildirParser(maildir_path)
        
        # Open specific folder
        if parser.open_folder(folder_name):
            print(f"Successfully opened folder: {folder_name}")
            
            # Get email count
            count = parser.get_email_count()
            print(f"Total emails in {folder_name}: {count}")
            
            # Get first few emails
            emails = parser.get_all_emails()
            print(f"\nFirst 3 emails in {folder_name}:")
            for i, email in enumerate(emails[:3]):
                print(f"  {i+1}. Subject: {email['subject']}")
                print(f"     From: {email['from']}")
                print(f"     Date: {email['date']}")
                print()
        else:
            print(f"Could not open folder: {folder_name}")
        
        # Clean up
        parser.close_mailbox()
        
    except Exception as e:
        print(f"Error: {e}")


def example_search_across_folders(maildir_path: str, search_term: str):
    """Example of searching across all folders."""
    print(f"=== Search Across All Folders Example: '{search_term}' ===")
    
    try:
        parser = MaildirParser(maildir_path)
        
        # Search across all folders
        matching_emails = parser.search_all_folders(subject=search_term)
        
        print(f"Found {len(matching_emails)} emails with '{search_term}' in subject across all folders:")
        
        # Group by folder
        emails_by_folder = {}
        for email in matching_emails:
            folder = email.get('folder', 'Unknown')
            if folder not in emails_by_folder:
                emails_by_folder[folder] = []
            emails_by_folder[folder].append(email)
        
        for folder, emails in emails_by_folder.items():
            print(f"\n  {folder} ({len(emails)} emails):")
            for email in emails:
                print(f"    - {email['subject']} (from: {email['from']})")
        
        # Clean up
        parser.close_mailbox()
        
    except Exception as e:
        print(f"Error: {e}")


def example_compare_folders(maildir_path: str):
    """Example of comparing email statistics across folders."""
    print("=== Folder Comparison Example ===")
    
    try:
        parser = MaildirParser(maildir_path)
        
        # Get info for all folders
        folders_info = parser.get_all_folders_info()
        
        print("Email statistics by folder:")
        print("-" * 50)
        
        total_emails = 0
        for folder_name, info in folders_info.items():
            count = info['email_count']
            total_emails += count
            print(f"{folder_name:15}: {count:4d} emails")
        
        print("-" * 50)
        print(f"{'TOTAL':15}: {total_emails:4d} emails")
        
        # Find folder with most emails
        if folders_info:
            largest_folder = max(folders_info.items(), key=lambda x: x[1]['email_count'])
            print(f"\nLargest folder: {largest_folder[0]} ({largest_folder[1]['email_count']} emails)")
        
        # Clean up
        parser.close_mailbox()
        
    except Exception as e:
        print(f"Error: {e}")


def example_export_folder_to_json(maildir_path: str, folder_name: str, output_file: str):
    """Example of exporting a specific folder to JSON."""
    print(f"=== Export Folder to JSON Example: {folder_name} ===")
    
    try:
        parser = MaildirParser(maildir_path)
        
        # Get emails from specific folder
        emails = parser.get_emails_from_folder(folder_name)
        
        # Export to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(emails, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Exported {len(emails)} emails from {folder_name} to {output_file}")
        
        # Clean up
        parser.close_mailbox()
        
    except Exception as e:
        print(f"Error: {e}")


def example_attachment_analysis_by_folder(maildir_path: str):
    """Example of analyzing attachments across different folders."""
    print("=== Attachment Analysis by Folder Example ===")
    
    try:
        parser = MaildirParser(maildir_path)
        
        folders_info = parser.get_all_folders_info()
        
        print("Attachment analysis by folder:")
        print("-" * 60)
        
        for folder_name in folders_info.keys():
            if parser.open_folder(folder_name):
                emails = parser.get_all_emails()
                
                # Count attachments
                total_attachments = 0
                attachment_types = {}
                
                for email in emails:
                    for attachment in email['attachments']:
                        total_attachments += 1
                        content_type = attachment['content_type']
                        attachment_types[content_type] = attachment_types.get(content_type, 0) + 1
                
                print(f"{folder_name:15}: {total_attachments:3d} attachments")
                
                # Show top attachment types for this folder
                if attachment_types:
                    top_types = sorted(attachment_types.items(), key=lambda x: x[1], reverse=True)[:3]
                    for content_type, count in top_types:
                        print(f"                {content_type}: {count}")
                print()
        
        # Clean up
        parser.close_mailbox()
        
    except Exception as e:
        print(f"Error: {e}")


def example_sender_analysis_by_folder(maildir_path: str):
    """Example of analyzing email senders by folder."""
    print("=== Sender Analysis by Folder Example ===")
    
    try:
        parser = MaildirParser(maildir_path)
        
        folders_info = parser.get_all_folders_info()
        
        print("Top senders by folder:")
        print("-" * 60)
        
        for folder_name in folders_info.keys():
            if parser.open_folder(folder_name):
                emails = parser.get_all_emails()
                
                # Count emails by sender
                sender_counts = {}
                for email in emails:
                    sender = email['from']
                    if sender:
                        sender_counts[sender] = sender_counts.get(sender, 0) + 1
                
                # Show top 3 senders for this folder
                if sender_counts:
                    top_senders = sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"{folder_name:15}:")
                    for sender, count in top_senders:
                        print(f"                {sender}: {count} emails")
                    print()
        
        # Clean up
        parser.close_mailbox()
        
    except Exception as e:
        print(f"Error: {e}")


def example_html_conversion(maildir_path: str, folder_name: str):
    """Example of HTML to plain text conversion."""
    print(f"=== HTML Conversion Example: {folder_name} ===")
    
    try:
        # Parser with HTML conversion enabled (default)
        parser_with_conversion = MaildirParser(maildir_path, convert_html=True)
        
        # Parser with aggressive cleaning
        parser_aggressive = MaildirParser(maildir_path, convert_html=True, aggressive_clean=True)
        
        # Parser with HTML conversion disabled
        parser_no_conversion = MaildirParser(maildir_path, convert_html=False)
        
        # Get emails with HTML conversion
        if parser_with_conversion.open_folder(folder_name):
            emails_with_conversion = parser_with_conversion.get_all_emails()
            
            # Find HTML emails
            html_emails = [e for e in emails_with_conversion if 'HTML' in e.get('content_type', '')]
            
            if html_emails:
                print(f"Found {len(html_emails)} HTML emails in {folder_name}")
                
                # Show first HTML email with different conversion methods
                email = html_emails[0]
                print(f"\nEmail: {email['subject']}")
                
                # Standard conversion
                print(f"  Standard conversion: {email['content_type']}")
                print(f"  Body Preview: {email['body'][:200]}...")
                
                # Aggressive cleaning
                if parser_aggressive.open_folder(folder_name):
                    emails_aggressive = parser_aggressive.get_all_emails()
                    aggressive_email = next((e for e in emails_aggressive if e['subject'] == email['subject']), None)
                    if aggressive_email:
                        print(f"\n  Aggressive cleaning: {aggressive_email['content_type']}")
                        print(f"  Body Preview: {aggressive_email['body'][:200]}...")
                
                # Show the same email without conversion
                if parser_no_conversion.open_folder(folder_name):
                    emails_no_conversion = parser_no_conversion.get_all_emails()
                    # Find matching email by subject
                    matching_email = next((e for e in emails_no_conversion if e['subject'] == email['subject']), None)
                    if matching_email:
                        print(f"\n  HTML preserved: {matching_email['content_type']}")
                        print(f"  Body Preview: {matching_email['body'][:200]}...")
            else:
                print(f"No HTML emails found in {folder_name}")
        
        # Clean up
        parser_with_conversion.close_mailbox()
        parser_aggressive.close_mailbox()
        parser_no_conversion.close_mailbox()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Example usage - replace with your actual Maildir path
    MAILDIR_PATH = "/path/to/your/maildir"  # Change this to your actual path
    
    print("Maildir Parser Examples - Multi-Folder Support")
    print("=" * 60)
    print(f"Using Maildir path: {MAILDIR_PATH}")
    print()
    
    # Uncomment and modify these examples as needed:
    
    # example_list_folders(MAILDIR_PATH)
    # example_process_specific_folder(MAILDIR_PATH, "INBOX")
    # example_process_specific_folder(MAILDIR_PATH, "Drafts")
    # example_search_across_folders(MAILDIR_PATH, "meeting")
    # example_compare_folders(MAILDIR_PATH)
    # example_export_folder_to_json(MAILDIR_PATH, "INBOX", "inbox_emails.json")
    # example_attachment_analysis_by_folder(MAILDIR_PATH)
    # example_sender_analysis_by_folder(MAILDIR_PATH)
    # example_html_conversion(MAILDIR_PATH, "INBOX")
    
    print("\nTo run examples, edit this file and uncomment the desired examples.")
    print("Make sure to set the correct MAILDIR_PATH variable.")
    print("\nAvailable examples:")
    print("  - List all folders and email counts")
    print("  - Process specific folders (INBOX, Drafts, etc.)")
    print("  - Search across all folders")
    print("  - Compare folder statistics")
    print("  - Export specific folders to JSON")
    print("  - Analyze attachments by folder")
    print("  - Analyze senders by folder")
    print("  - HTML to plain text conversion")
