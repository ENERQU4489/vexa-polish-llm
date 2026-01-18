"""
WikiDownloader - Automatic downloader for Polish Wikipedia articles
Uses Wikipedia API to download article content
"""

import requests
import time
import os
from typing import List, Optional, Dict
from tqdm import tqdm


class WikiDownloader:
    """
    Class for downloading articles from Polish Wikipedia.
    Uses MediaWiki API.
    """
    
    def __init__(self, language: str = 'pl', output_dir: str = 'data/raw'):
        """
        Initialize downloader.
        
        Args:
            language: Wikipedia language code (default 'pl')
            output_dir: Output directory for downloaded files
        """
        self.language = language
        self.output_dir = output_dir
        self.api_url = f'https://{language}.wikipedia.org/w/api.php'
        self.session = requests.Session()
        
        self.session.headers.update({
            'User-Agent': 'VexaPolishLLM/1.0 (Educational Project; Python/requests)'
        })
        
        os.makedirs(output_dir, exist_ok=True)
    
    def get_random_articles(self, count: int = 10) -> List[Dict[str, str]]:
        """
        Download random articles from Wikipedia.
        
        Args:
            count: Number of articles to download
            
        Returns:
            List of dictionaries with article titles and content
        """
        articles = []
        
        print(f"Downloading {count} random articles from {self.language}.wikipedia.org...")
        
        for _ in tqdm(range(count), desc="Downloading articles"):
            try:
                params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'random',
                    'rnnamespace': 0,
                    'rnlimit': 1
                }
                
                response = self.session.get(self.api_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'query' in data and 'random' in data['query']:
                    title = data['query']['random'][0]['title']
                    
                    content = self.get_article_content(title)
                    
                    if content:
                        articles.append({
                            'title': title,
                            'content': content
                        })
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n⚠ Error downloading article: {e}")
                continue
        
        print(f"✓ Downloaded {len(articles)} articles")
        return articles
    
    def get_article_content(self, title: str) -> Optional[str]:
        """
        Download content of specific article.
        
        Args:
            title: Article title
            
        Returns:
            Article content or None on error
        """
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'explaintext': True,
                'exsectionformat': 'plain'
            }
            
            response = self.session.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            
            if page_id != '-1' and 'extract' in pages[page_id]:
                return pages[page_id]['extract']
            
            return None
            
        except Exception as e:
            print(f"⚠ Error downloading '{title}': {e}")
            return None
    
    def get_articles_by_category(self, category: str, count: int = 10) -> List[Dict[str, str]]:
        """
        Download articles from specific category.
        
        Args:
            category: Category name (e.g., "History of Poland")
            count: Number of articles to download
            
        Returns:
            List of dictionaries with article titles and content
        """
        articles = []
        
        print(f"Downloading articles from category '{category}'...")
        
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': count,
                'cmtype': 'page'
            }
            
            response = self.session.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'categorymembers' in data['query']:
                members = data['query']['categorymembers']
                
                for member in tqdm(members[:count], desc="Downloading articles"):
                    title = member['title']
                    content = self.get_article_content(title)
                    
                    if content:
                        articles.append({
                            'title': title,
                            'content': content
                        })
                    
                    time.sleep(0.5)
        
        except Exception as e:
            print(f"⚠ Error downloading category: {e}")
        
        print(f"✓ Downloaded {len(articles)} articles from category")
        return articles
    
    def search_articles(self, query: str, count: int = 10) -> List[Dict[str, str]]:
        """
        Search articles by query.
        
        Args:
            query: Search query
            count: Number of articles to download
            
        Returns:
            List of dictionaries with article titles and content
        """
        articles = []
        
        print(f"Searching articles: '{query}'...")
        
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': count,
                'srnamespace': 0
            }
            
            response = self.session.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'search' in data['query']:
                results = data['query']['search']
                
                for result in tqdm(results[:count], desc="Downloading articles"):
                    title = result['title']
                    content = self.get_article_content(title)
                    
                    if content:
                        articles.append({
                            'title': title,
                            'content': content
                        })
                    
                    time.sleep(0.5)
        
        except Exception as e:
            print(f"⚠ Error searching: {e}")
        
        print(f"✓ Found {len(articles)} articles")
        return articles
    
    def save_articles(self, articles: List[Dict[str, str]], prefix: str = 'wiki') -> None:
        """
        Save articles to text files.
        
        Args:
            articles: List of articles to save
            prefix: Filename prefix
        """
        print(f"Saving {len(articles)} articles to {self.output_dir}...")
        
        for idx, article in enumerate(tqdm(articles, desc="Saving")):
            filename = f"{prefix}_{idx:04d}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"TITLE: {article['title']}\n\n")
                    f.write(article['content'])
                
            except Exception as e:
                print(f"\n⚠ Error saving {filename}: {e}")
        
        print(f"✓ Articles saved in {self.output_dir}")
    
    def download_and_save(self, count: int = 100, method: str = 'random') -> None:
        """
        Download and save articles (all-in-one method).
        
        Args:
            count: Number of articles to download
            method: Download method ('random', 'category', 'search')
        """
        if method == 'random':
            articles = self.get_random_articles(count)
        else:
            print(f"⚠ Unknown method: {method}. Using 'random'.")
            articles = self.get_random_articles(count)
        
        if articles:
            self.save_articles(articles)
        else:
            print("⚠ No articles downloaded")
