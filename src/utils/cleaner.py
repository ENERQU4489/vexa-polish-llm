"""
TextCleaner - Tool for text cleaning and normalization
Optimized for Polish language
"""

import re
from typing import List


class TextCleaner:
    """
    Class for cleaning and normalizing Polish language texts.
    Removes unnecessary characters, normalizes whitespace, etc.
    """
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_numbers: bool = False,
                 lowercase: bool = False,
                 remove_extra_whitespace: bool = True):
        """
        Initialize cleaner.
        
        Args:
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_numbers: Whether to remove numbers
            lowercase: Whether to convert to lowercase
            remove_extra_whitespace: Whether to remove extra whitespace
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
        
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
        
        self.polish_chars = 'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'
    
    def clean(self, text: str) -> str:
        """
        Clean single text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_extra_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
            text = text.strip()
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean list of texts.
        
        Args:
            texts: List of texts to clean
            
        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]
    
    def remove_wiki_markup(self, text: str) -> str:
        """
        Remove Wiki markup from text.
        
        Args:
            text: Text with Wiki markup
            
        Returns:
            Text without markup
        """
        text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]+)\]\]', r'\1', text)
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*\/>', '', text)
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        text = re.sub(r'={2,}[^=]+={2,}', '', text)
        text = re.sub(r'^\*+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        
        return text
    
    def normalize_polish_text(self, text: str) -> str:
        """
        Normalize Polish text (preserves Polish characters).
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        allowed_chars = f'a-zA-Z{self.polish_chars}0-9\\s.,!?;:()\"-'
        pattern = f'[^{allowed_chars}]'
        text = re.sub(pattern, ' ', text)
        
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        
        text = self.whitespace_pattern.sub(' ', text)
        text = text.strip()
        
        return text
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences (simple splitter).
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def remove_short_texts(self, texts: List[str], min_length: int = 10) -> List[str]:
        """
        Remove texts that are too short.
        
        Args:
            texts: List of texts
            min_length: Minimum text length
            
        Returns:
            List of texts longer than min_length
        """
        return [text for text in texts if len(text) >= min_length]
    
    def deduplicate(self, texts: List[str]) -> List[str]:
        """
        Remove duplicates from list of texts (preserves order).
        
        Args:
            texts: List of texts
            
        Returns:
            List of unique texts
        """
        seen = set()
        unique_texts = []
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        return unique_texts
