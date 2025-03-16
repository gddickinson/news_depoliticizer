import requests
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file (for API keys)
load_dotenv()

class NewsAggregator:
    def __init__(self):
        # API keys stored in environment variables for security
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.nyt_key = os.getenv("NYT_KEY")
        self.guardian_key = os.getenv("GUARDIAN_KEY")
        
        # Store articles from different sources
        self.articles = {
            "newsapi": [],
            "nyt": [],
            "guardian": []
        }
        
        # For story similarity matching
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def fetch_newsapi_headlines(self, category='general', country='us', page_size=30):
        """Fetch headlines from NewsAPI"""
        url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&pageSize={page_size}&apiKey={self.newsapi_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if data['status'] == 'ok':
                # Get articles and add source info
                articles = data['articles']
                for i, article in enumerate(articles):
                    article['prominence_score'] = page_size - i  # Higher score for earlier articles
                    article['source_name'] = 'newsapi'
                
                self.articles['newsapi'] = articles
                print(f"Retrieved {len(articles)} articles from NewsAPI")
            else:
                print(f"Error fetching from NewsAPI: {data.get('message', 'Unknown error')}")
        
        except Exception as e:
            print(f"Exception occurred with NewsAPI: {str(e)}")
    
    def fetch_nyt_headlines(self, section='home', num_results=30):
        """Fetch headlines from New York Times API"""
        url = f"https://api.nytimes.com/svc/topstories/v2/{section}.json?api-key={self.nyt_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'results' in data:
                articles = data['results']
                for i, article in enumerate(articles[:num_results]):
                    # Standardize the format to match our structure
                    formatted_article = {
                        'title': article.get('title', ''),
                        'description': article.get('abstract', ''),
                        'url': article.get('url', ''),
                        'publishedAt': article.get('published_date', ''),
                        'content': article.get('abstract', ''),  # NYT API doesn't provide full content
                        'prominence_score': num_results - i,
                        'source_name': 'nyt'
                    }
                    self.articles['nyt'].append(formatted_article)
                
                print(f"Retrieved {len(self.articles['nyt'])} articles from NYT")
            else:
                print(f"Error fetching from NYT API: {data.get('message', 'Unknown error')}")
        
        except Exception as e:
            print(f"Exception occurred with NYT API: {str(e)}")
    
    def fetch_guardian_headlines(self, section='world', page_size=30):
        """Fetch headlines from Guardian API"""
        url = f"https://content.guardianapis.com/{section}?api-key={self.guardian_key}&show-fields=headline,trailText,body,publication&page-size={page_size}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'response' in data and 'results' in data['response']:
                articles = data['response']['results']
                for i, article in enumerate(articles):
                    # Standardize the format
                    formatted_article = {
                        'title': article.get('webTitle', ''),
                        'description': article.get('fields', {}).get('trailText', ''),
                        'url': article.get('webUrl', ''),
                        'publishedAt': article.get('webPublicationDate', ''),
                        'content': article.get('fields', {}).get('body', ''),
                        'prominence_score': page_size - i,
                        'source_name': 'guardian'
                    }
                    self.articles['guardian'].append(formatted_article)
                
                print(f"Retrieved {len(self.articles['guardian'])} articles from Guardian")
            else:
                print(f"Error fetching from Guardian API: {data.get('fault', 'Unknown error')}")
        
        except Exception as e:
            print(f"Exception occurred with Guardian API: {str(e)}")
    
    def fetch_all_headlines(self):
        """Fetch headlines from all configured sources"""
        self.fetch_newsapi_headlines()
        self.fetch_nyt_headlines()
        self.fetch_guardian_headlines()
        
        return self.get_all_articles()
    
    def get_all_articles(self):
        """Combine all articles into a single list"""
        all_articles = []
        for source, articles in self.articles.items():
            all_articles.extend(articles)
        
        return all_articles
    
    def identify_similar_stories(self, threshold=0.6):
        """Group similar stories across sources based on title and description similarity"""
        all_articles = self.get_all_articles()
        
        # If no articles, return empty list
        if not all_articles:
            return []
        
        # Extract titles and descriptions for similarity comparison
        docs = []
        for article in all_articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            docs.append(text)
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(docs)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Group similar articles
        story_groups = []
        processed_indices = set()
        
        for i in range(len(all_articles)):
            if i in processed_indices:
                continue
                
            # Find all similar articles above the threshold
            similar_indices = [i]
            for j in range(len(all_articles)):
                if i != j and j not in processed_indices and similarity_matrix[i, j] > threshold:
                    similar_indices.append(j)
            
            # Create story group
            group = {
                'articles': [all_articles[idx] for idx in similar_indices],
                'sources': list(set(all_articles[idx]['source_name'] for idx in similar_indices)),
                'avg_prominence': np.mean([all_articles[idx]['prominence_score'] for idx in similar_indices]),
                'main_title': all_articles[i]['title'],  # Use the first article's title as main
                'article_count': len(similar_indices)
            }
            
            story_groups.append(group)
            processed_indices.update(similar_indices)
        
        # Add remaining articles as individual stories
        for i in range(len(all_articles)):
            if i not in processed_indices:
                group = {
                    'articles': [all_articles[i]],
                    'sources': [all_articles[i]['source_name']],
                    'avg_prominence': all_articles[i]['prominence_score'],
                    'main_title': all_articles[i]['title'],
                    'article_count': 1
                }
                story_groups.append(group)
        
        # Sort by average prominence score (higher = more important)
        story_groups.sort(key=lambda x: x['avg_prominence'], reverse=True)
        
        return story_groups
    
    def prepare_for_llm_rewrite(self, story_groups):
        """Prepare stories for LLM rewriting by extracting key information"""
        llm_input_data = []
        
        for group in story_groups:
            # Combine content from all articles in the group
            combined_content = ""
            source_details = []
            
            for article in group['articles']:
                content = article.get('content', article.get('description', ''))
                if content:
                    combined_content += content + "\n\n"
                
                # Get more detailed source information when available
                source_name = article.get('source_name', 'unknown')
                if source_name == 'newsapi' and article.get('source', {}).get('name'):
                    # If we have the actual publication name from NewsAPI, use it
                    detailed_source = article['source']['name']
                    source_details.append(detailed_source)
                elif source_name not in ['unknown', 'newsapi']:
                    source_details.append(source_name)
            
            # If we don't have detailed sources, use the original sources list
            if not source_details:
                source_details = group['sources']
            
            # Remove duplicates while preserving order
            seen = set()
            unique_sources = [x for x in source_details if not (x in seen or seen.add(x))]
            
            llm_data = {
                'title': group['main_title'],
                'content': combined_content,
                'sources': unique_sources,  # Use more detailed source names
                'original_sources': group['sources'],  # Keep original sources as backup
                'importance_rank': len(story_groups) - story_groups.index(group),
                'total_sources': len(group['sources'])
            }
            
            llm_input_data.append(llm_data)
        
        return llm_input_data

    def save_to_json(self, data, filename="news_data.json"):
        """Save data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {filename}")


# Example LLM prompt template for rewriting (to be used with your local LLM)
def create_llm_prompt(news_item):
    prompt = f"""
    Rewrite the following news story in a factual, balanced manner removing political bias and inflammatory language.
    
    Title: {news_item['title']}
    
    Original Content:
    {news_item['content'][:2000]}  # Truncate if too long for your model
    
    This story appears in {news_item['total_sources']} news sources and is ranked #{news_item['importance_rank']} in importance.
    
    Provide a rewritten version that:
    1. Maintains all factual information
    2. Removes political bias or charged language
    3. Presents multiple perspectives if applicable
    4. Uses neutral, objective language
    """
    return prompt


if __name__ == "__main__":
    # Example usage
    aggregator = NewsAggregator()
    
    # Fetch headlines from all sources
    aggregator.fetch_all_headlines()
    
    # Group similar stories
    story_groups = aggregator.identify_similar_stories(threshold=0.6)
    print(f"Identified {len(story_groups)} distinct story groups")
    
    # Prepare for LLM rewriting
    llm_input_data = aggregator.prepare_for_llm_rewrite(story_groups)
    
    # Save data for LLM processing
    aggregator.save_to_json(llm_input_data, "llm_input_data.json")
    
    # Example of generating a prompt for the first story
    if llm_input_data:
        example_prompt = create_llm_prompt(llm_input_data[0])
        print("\nExample LLM Prompt for first story:")
        print(example_prompt)
