import requests
import os
from typing import Union, Sequence, List, Dict
from datetime import datetime
from datetime import datetime, timedelta
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import RecursiveUrlLoader
from RAG.deep_article_scraper import DeepArticleScraper, ArticleContent
import logging
from bs4 import BeautifulSoup as Soup

logger = logging.getLogger(__name__)

class TavilyController:
    def __init__(self, enable_deep_scraping: bool = False):
        """
        Initialize TavilyController with optional deep scraping

        Args:
            enable_deep_scraping: If True, enables intelligent deep article scraping
        """
        self.__api_key__ = os.getenv("Tavily_API_KEY")
        self.__base_url__ = "https://api.tavily.com"
        self.__tavily_search_url__ ="https://api.tavily.com/search"
        self.__model__ = "gpt-5-nano"
        self.__chat_model_instance__ = init_chat_model(model=self.__model__,
                                         model_provider="openai",
                                         temperature=0)
        self.__tool__  = TavilySearch(
            max_results=5,
            topic= "general",
            # include_answer=False,
            include_raw_content=True,
            # include_images=False,
            # include_image_descriptions=False,
            # search_depth="basic",
            # time_range="day",
            # include_domains=None,
            # exclude_domains=None
        )

        # Initialize deep scraper if enabled
        self.enable_deep_scraping = enable_deep_scraping
        self.deep_scraper = DeepArticleScraper() if enable_deep_scraping else None


    def agent_search(self, query, max_results=10, topic="general"):
        """
        Search recent news using Tavily API and LangChain agent.

        Args:
            query (str): The search query.
            max_results (int): Maximum number of news articles to return.
            topic (str): Topic category for the search.

        Returns:
            list: List of news articles (dicts).
        """
        agent = create_agent(model=self.__model__, tools=[self.__tool__])
        
        results : list = []
        for step in agent.stream(
            input={"messages" : query},
            stream_mode="values"
        ):
            results.append(step["messages"][-1])

        return results
        
        

    def _get_raw_content_(self, url):
        try:
            loader = RecursiveUrlLoader(url=url, 
                                        max_depth=1,
                                        extractor = lambda x : Soup(x, "html.parser").get_text())
        except Exception as e:
            logger.error(f"Error initializing RecursiveUrlLoader for {url}: {e}")

        docs = loader.load()
        if len(docs) ==  1:                
            return docs[0].page_content
        else:
            logger.warning(f"Unexpected number of documents extracted for {article.get('url')}: {len(docs)}")


    def search_by_query(self,
                        query,
                        topic="general",
                        domain_list : Union[Sequence[str], str] = None,
                        deep_scrape: bool = None):
        """
        Search using Tavily API with optional deep scraping

        Args:
            query: Search query string
            topic: Topic category for search ("general", "news", etc.)
            domain_list: List of domains to include in search
            deep_scrape: If True, performs deep article scraping. If None, uses instance default.

        Returns:
            List of article dictionaries with title, content, published date, and URL
            If deep_scrape is enabled, content will be full-text instead of snippets
        """
        client = TavilyClient(api_key=self.__api_key__)
        domain_list = domain_list if type(domain_list) is list else [domain_list] if domain_list else None

        try:
            response = client.search(
                query=query,
                topic=topic,
                include_domains=domain_list
            )
        except:
            response = {"results": []}

        # Get basic results
        basic_results = [
            dict(
                title=r.get('title'),
                content=r.get('content'),
                published=r.get('published_date'),
                raw_content= self._get_raw_content_(r.get('url')),
                summary=r.get('summary'),
                url=r.get('url')
            )
            for r in response.get('results')
        ]

        # Determine if we should deep scrape
        #should_deep_scrape = deep_scrape if deep_scrape is not None else self.enable_deep_scraping
        # if should_deep_scrape and self.deep_scraper:
        #     logger.info(f"Deep scraping enabled for {len(basic_results)} results")
        #     enhanced_deep_results =  self._enhance_with_deep_scraping(basic_results)
        #     basic_results.extend(enhanced_deep_results)

        return basic_results

    def _enhance_with_deep_scraping(self, tavily_results: List[Dict]) -> List[Dict]:
        """
        Enhance Tavily results with deep scraping

        Args:
            tavily_results: Basic results from Tavily
        Returns:
            Enhanced results with full-text content
        """
        enhanced_results = []

        # Use DeepArticleScraper to get full content
        full_articles: List[ArticleContent] = self.deep_scraper.scrape_tavily_results(tavily_results)

        # Convert ArticleContent objects back to dict format
        for article in full_articles:
            enhanced_results.append({
                'title': article.title,
                'content': article.full_text if article.full_text else article.summary,
                'summary': article.summary,
                'published': article.published_date,
                'url': article.url,
                'author': article.author,
                'word_count': article.word_count,
                'images': article.images,
                'source_domain': article.source_domain,
                'extraction_timestamp': article.extraction_timestamp,
                'full_text': article.full_text  # Full text for LLM processing
            })
            

        #logger.info(f"Deep scraping complete: {len(enhanced_results)} articles with full text")
        return enhanced_results

    def extract_content(self,
                       urls: Union[str, List[str]],
                       query: str = None,
                       max_chunks: int = 3,
                       extract_depth: str = "basic") -> List[Dict]:
        """
        Extract content from URLs using Tavily Extract API

        Args:
            urls: Single URL string or list of URLs to extract content from
            query: Optional query for reranking extracted chunks by relevance
            max_chunks: Number of content chunks to extract per URL (1-5, default: 3)
            extract_depth: Extraction depth - "basic" (faster) or "advanced" (more comprehensive)

        Returns:
            List of dictionaries containing extracted content:
            [
                {
                    'url': str,
                    'raw_content': str,  # Full extracted text
                    'chunks': List[str],  # Content chunks (max 500 chars each)
                    'success': bool,
                    'error': str or None
                }
            ]

        Note:
            - basic extraction: 1 credit per 5 successful URLs
            - advanced extraction: 2 credits per 5 successful URLs (includes tables, embedded content)
            - Chunks are ranked by relevance if query is provided
        """
        client = TavilyClient(api_key=self.__api_key__)

        # Normalize URLs to list
        url_list = [urls] if isinstance(urls, str) else urls

        # Validate parameters
        max_chunks = max(1, min(5, max_chunks))  # Clamp between 1-5
        if extract_depth not in ["basic", "advanced"]:
            logger.warning(f"Invalid extract_depth '{extract_depth}', defaulting to 'basic'")
            extract_depth = "basic"

        try:
            # Call Tavily Extract API
            response = client.extract(
                urls=url_list,
                query=query,
                max_results=max_chunks,
                depth=extract_depth
            )

            # Parse response
            results = []
            successful_extractions = response.get('results', [])
            failed_urls = response.get('failed_results', [])

            # Process successful extractions
            for extraction in successful_extractions:
                results.append({
                    'url': extraction.get('url'),
                    'raw_content': extraction.get('raw_content', ''),
                    'chunks': extraction.get('results', []),  # List of content chunks
                    'success': True,
                    'error': None
                })

            # Process failed extractions
            for failed_url in failed_urls:
                results.append({
                    'url': failed_url,
                    'raw_content': '',
                    'chunks': [],
                    'success': False,
                    'error': 'Extraction failed'
                })

            logger.info(f"Extracted content from {len(successful_extractions)}/{len(url_list)} URLs")
            return results

        except Exception as e:
            logger.error(f"Tavily extract API error: {e}")
            # Return error result for all URLs
            return [
                {
                    'url': url,
                    'raw_content': '',
                    'chunks': [],
                    'success': False,
                    'error': str(e)
                }
                for url in url_list
            ]

# Example usage:
# stock = "IONQ"
# query =f"What are the latest updates with {stock}"
# tavily = TavilyController()
# news = tavily.search_stock_news(query)
# #news= tavily.search_by_query()
# for article in news:
#     print(f"{article['title']} - {article['content']}\n")