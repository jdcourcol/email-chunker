import os
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import PyPDF2
import magic
from pathlib import Path


class PDFParser:
    """Parse PDF documents and extract text content for search indexing."""
    
    def __init__(self):
        self.supported_extensions = {'.pdf'}
        self.mime_types = {'application/pdf'}
    
    def is_pdf_file(self, file_path: str) -> bool:
        """Check if a file is a PDF based on extension and MIME type."""
        try:
            # Check file extension
            if not any(file_path.lower().endswith(ext) for ext in self.supported_extensions):
                return False
            
            # Check MIME type
            mime_type = magic.from_file(file_path, mime=True)
            return mime_type in self.mime_types
        except Exception:
            return False
    
    def parse_pdf(self, file_path: str) -> Dict:
        """Parse a PDF file and extract metadata and content."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = pdf_reader.metadata or {}
                
                # Extract text from all pages
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"Page {page_num + 1}: {page_text.strip()}")
                    except Exception as e:
                        text_content.append(f"Page {page_num + 1}: [Error extracting text: {str(e)}]")
                
                # Combine all text
                full_text = "\n\n".join(text_content)
                
                # Clean the text
                cleaned_text = self._clean_text(full_text)
                
                # Get file info
                file_stat = os.stat(file_path)
                file_path_obj = Path(file_path)
                
                return {
                    'file_path': file_path,
                    'file_name': file_path_obj.name,
                    'file_size': file_stat.st_size,
                    'created_date': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    'modified_date': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    'page_count': len(pdf_reader.pages),
                    'title': metadata.get('/Title', ''),
                    'author': metadata.get('/Author', ''),
                    'subject': metadata.get('/Subject', ''),
                    'creator': metadata.get('/Creator', ''),
                    'producer': metadata.get('/Producer', ''),
                    'content': cleaned_text,
                    'content_length': len(cleaned_text),
                    'file_type': 'pdf'
                }
                
        except Exception as e:
            return {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'error': str(e),
                'file_type': 'pdf'
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        if not text:
            return text
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+:', '', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\-.,!?;:()@#$%&*+=<>[\]{}|\\/~`"\'_–—…\n]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def get_pdf_files_in_directory(self, directory_path: str) -> List[str]:
        """Get all PDF files in a directory and subdirectories."""
        pdf_files = []
        
        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self.is_pdf_file(file_path):
                        pdf_files.append(file_path)
        except Exception as e:
            print(f"Error scanning directory {directory_path}: {e}")
        
        return sorted(pdf_files)
    
    def batch_parse_pdfs(self, directory_path: str, max_files: Optional[int] = None) -> List[Dict]:
        """Parse multiple PDF files in a directory."""
        pdf_files = self.get_pdf_files_in_directory(directory_path)
        
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        parsed_pdfs = []
        for pdf_file in pdf_files:
            try:
                parsed = self.parse_pdf(pdf_file)
                if 'error' not in parsed:
                    parsed_pdfs.append(parsed)
                    print(f"✓ Parsed: {parsed['file_name']} ({parsed['page_count']} pages)")
                else:
                    print(f"✗ Error parsing: {parsed['file_name']} - {parsed['error']}")
            except Exception as e:
                print(f"✗ Failed to parse {pdf_file}: {e}")
        
        return parsed_pdfs


class PDFSearchIndexer:
    """Index PDF documents for search functionality."""
    
    def __init__(self, db_manager, embedding_model):
        self.db_manager = db_manager
        self.embedding_model = embedding_model
        self.pdf_parser = PDFParser()
    
    def index_pdf_documents(self, directory_path: str, folder_name: str = "documents") -> Dict:
        """Index all PDF documents in a directory to the database."""
        print(f"Scanning for PDF documents in: {directory_path}")
        
        # Parse all PDFs
        parsed_pdfs = self.pdf_parser.batch_parse_pdfs(directory_path)
        
        if not parsed_pdfs:
            print("No PDF documents found to index.")
            return {'total': 0, 'indexed': 0, 'errors': 0}
        
        print(f"Found {len(parsed_pdfs)} PDF documents to index.")
        
        indexed_count = 0
        error_count = 0
        
        for pdf_doc in parsed_pdfs:
            try:
                # Generate embedding for the content
                if pdf_doc['content'] and len(pdf_doc['content']) > 10:
                    embedding = self.embedding_model.encode(pdf_doc['content']).tolist()
                    
                    # Save to database
                    success = self.db_manager.save_pdf_document(
                        file_path=pdf_doc['file_path'],
                        file_name=pdf_doc['file_name'],
                        folder_name=folder_name,
                        content=pdf_doc['content'],
                        metadata=pdf_doc,
                        embedding=embedding
                    )
                    
                    if success:
                        indexed_count += 1
                        print(f"✓ Indexed: {pdf_doc['file_name']}")
                    else:
                        error_count += 1
                        print(f"✗ Failed to save: {pdf_doc['file_name']}")
                else:
                    print(f"⚠ Skipped (no content): {pdf_doc['file_name']}")
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"✗ Error indexing {pdf_doc['file_name']}: {e}")
        
        print(f"\nIndexing complete: {indexed_count} indexed, {error_count} errors")
        return {
            'total': len(parsed_pdfs),
            'indexed': indexed_count,
            'errors': error_count
        }
    
    def search_pdf_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search through indexed PDF documents."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in database
            results = self.db_manager.search_pdf_documents(
                query=query,
                query_embedding=query_embedding,
                limit=limit
            )
            
            return results
            
        except Exception as e:
            print(f"Error searching PDF documents: {e}")
            return []
