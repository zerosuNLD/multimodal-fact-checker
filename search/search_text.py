import requests
import json
from setting import serper_api_key
from .exclude import EXCLUDE_DOMAINS

def search_text(query, num_results=5, SERPER_API_KEY=serper_api_key):
    url = "https://google.serper.dev/search"
    
    payload = json.dumps({
        "q": query,
        "num": max(10, num_results + 5), 
        "autostructured": True
    })
    
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    data = response.json()
    
    results = []
    if "organic" in data:
        for item in data["organic"]:
            link = item.get("link", "")
            
            is_excluded = any(domain in link for domain in EXCLUDE_DOMAINS)
            
            if is_excluded:
                continue
                
            results.append({
                "title": item.get("title"),
                "link": link,
                "snippet": item.get("snippet")
            })
            
            if len(results) >= num_results:
                break
                
    return results
