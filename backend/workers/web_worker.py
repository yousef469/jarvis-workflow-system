"""Web Worker V24 - Simple DuckDuckGo Search (No LangChain)"""

from typing import Dict, Any
import asyncio

class WebWorker:
    """Handles web search using DuckDuckGo directly (no LangChain)"""
    
    def __init__(self):
        self.name = "web"
        self._ddgs = None
    
    def _get_ddgs(self):
        """Lazy load DuckDuckGo search"""
        if self._ddgs is None:
            try:
                from duckduckgo_search import DDGS
                self._ddgs = DDGS()
            except ImportError:
                print("[WebWorker] Warning: duckduckgo-search not installed. Run: pip install duckduckgo-search")
                self._ddgs = None
        return self._ddgs
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """Execute web search task"""
        print(f"[WebWorker] 🌐 Searching for: {query}")
        
        ddgs = self._get_ddgs()
        if not ddgs:
            return {
                "source": "web",
                "error": "Search tool unavailable - duckduckgo-search not installed",
                "summary": "I'm unable to search the web at the moment, sir.",
                "status": "error"
            }
        
        try:
            # Perform search in thread to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: list(ddgs.text(query, max_results=5))
            )
            
            if not results:
                return {
                    "source": "web",
                    "query": query,
                    "summary": "No results found for that search.",
                    "status": "success"
                }
            
            # Format results into readable summary
            summaries = []
            for r in results[:5]:
                title = r.get("title", "")
                body = r.get("body", "")
                if title and body:
                    summaries.append(f"• {title}: {body[:200]}")
            
            summary = "\n".join(summaries)
            
            return {
                "source": "web",
                "query": query,
                "results_count": len(results),
                "summary": summary,
                "status": "success"
            }
            
        except Exception as e:
            print(f"[WebWorker] ❌ Search Failed: {e}")
            return {
                "source": "web",
                "error": str(e),
                "summary": f"I encountered an issue while searching: {str(e)[:100]}",
                "status": "error"
            }


# Singleton instance
web_worker = WebWorker()
