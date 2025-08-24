#!/usr/bin/env python3
"""
Cross-Encoder Reranker for Email Search

This module provides reranking functionality using cross-encoder models
to improve the relevance of search results.
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import numpy as np


class EmailReranker:
    """
    Reranker for email search results using cross-encoder models.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with a cross-encoder model.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            print(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            print(f"✅ Cross-encoder model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Failed to load cross-encoder model: {e}")
            print("   Reranking will be disabled")
            self.model = None
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], 
                      top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank search results using the cross-encoder model.
        
        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of top results to return (None = all)
            
        Returns:
            Reranked list of results with cross_encoder_score added
        """
        if not self.model or not results:
            return results
        
        try:
            # Prepare pairs for cross-encoder (query, document)
            pairs = []
            for result in results:
                # Create document text from relevant fields
                doc_text = self._create_document_text(result)
                pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Add scores to results
            for i, result in enumerate(results):
                result['cross_encoder_score'] = float(scores[i])
            
            # Sort by cross-encoder score (descending)
            results.sort(key=lambda x: x.get('cross_encoder_score', 0), reverse=True)
            
            # Return top_k results if specified
            if top_k is not None:
                return results[:top_k]
            
            return results
            
        except Exception as e:
            print(f"Error during reranking: {e}")
            return results
    
    def _create_document_text(self, result: Dict[str, Any]) -> str:
        """
        Create document text from email result for cross-encoder scoring.
        
        Args:
            result: Email result dictionary
            
        Returns:
            Formatted document text
        """
        parts = []
        
        # Add subject
        if result.get('subject'):
            parts.append(f"Subject: {result['subject']}")
        
        # Add sender
        if result.get('sender'):
            parts.append(f"From: {result['sender']}")
        
        # Add body (truncated if too long)
        if result.get('body'):
            body = result['body']
            if len(body) > 1000:  # Limit body length for cross-encoder
                body = body[:1000] + "..."
            parts.append(f"Content: {body}")
        
        return " | ".join(parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"status": "not_loaded", "model_name": self.model_name}
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "max_length": getattr(self.model, 'max_length', 'unknown')
        }


def create_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Optional[EmailReranker]:
    """
    Factory function to create a reranker instance.
    
    Args:
        model_name: Name of the cross-encoder model
        
    Returns:
        EmailReranker instance or None if failed
    """
    try:
        return EmailReranker(model_name)
    except Exception as e:
        print(f"Failed to create reranker: {e}")
        return None


if __name__ == "__main__":
    # Test the reranker
    reranker = create_reranker()
    if reranker:
        print("Reranker created successfully!")
        print("Model info:", reranker.get_model_info())
    else:
        print("Failed to create reranker")
