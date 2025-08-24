#!/usr/bin/env python3
"""
Test script for text cleaning functionality.
"""

import re

def clean_text(text: str) -> str:
    """
    Clean text by removing unwanted characters, pilcrows, and normalizing whitespace.
    
    Args:
        text: Text string to clean
        
    Returns:
        Cleaned text string
    """
    if not text:
        return text
    
    # Remove pilcrows and other unwanted characters
    text = re.sub(r'¶', '', text)  # Remove pilcrow symbols
    text = re.sub(r'[^\w\s\-.,!?;:()@#$%&*+=<>[\]{}|\\/~`"\'_–—…]', '', text)
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def test_text_cleaning():
    """Test various text cleaning scenarios."""
    
    test_cases = [
        # Test pilcrow removal
        ("Hello¶ World", "Hello World"),
        ("Subject¶: Test Email", "Subject: Test Email"),
        ("From¶: sender@example.com", "From: sender@example.com"),
        
        # Test other unwanted characters
        ("Hello\x00World", "HelloWorld"),
        ("Test\x01\x02\x03", "Test"),
        
        # Test whitespace normalization
        ("  Multiple    spaces  ", "Multiple spaces"),
        ("Line1\n\n\nLine2", "Line1\n\nLine2"),
        
        # Test mixed cases
        ("¶Subject¶: ¶Test¶ Email¶", "Subject: Test Email"),
        ("From¶: ¶sender¶@example.com¶", "From: sender@example.com"),
        
        # Test real email subject examples
        ("=?utf-8?B?RHJpZXMgc2VudCBhIG1lc3NhZ2U=?=", "utf-8BDRpZXMgc2VudCBhIG1lc3NhZ2U"),
        ("=?utf-8?B?RHJpZXMgVmVyYWNodGVydCBpbiBUZWFtcw==?=", "utf-8BDRpZXMgVmVyYWNodGVydCBpbiBUZWFtcw"),
    ]
    
    print("Testing Text Cleaning Function")
    print("=" * 50)
    
    all_passed = True
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        cleaned = clean_text(input_text)
        passed = cleaned == expected
        
        print(f"Test {i}:")
        print(f"  Input:    '{input_text}'")
        print(f"  Expected:  '{expected}'")
        print(f"  Got:       '{cleaned}'")
        print(f"  Status:    {'✅ PASS' if passed else '❌ FAIL'}")
        print()
        
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    test_text_cleaning()
