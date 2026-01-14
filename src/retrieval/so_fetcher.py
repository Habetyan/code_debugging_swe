"""
StackOverflow Fetcher: interacts with StackExchange API.
Fetches questions/answers by tag and converts them into retrieval documents.
"""
import requests
import html
import time
from typing import List
from .corpus import Document

class StackOverflowFetcher:
    API_URL = "https://api.stackexchange.com/2.3/questions"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def fetch_questions(self, tag: str, pages: int = 1) -> List[Document]:
        """Fetching questions and answers for a given tag."""
        documents = []
        
        for page in range(1, pages + 1):
            try:
                params = {
                    "order": "desc",
                    "sort": "votes",
                    "tagged": tag,
                    "site": "stackoverflow",
                    "key": self.api_key,
                    "pagesize": 20,
                    "page": page,
                    "filter": "!9_bDDxJY5"  # With body and answers
                }
                
                resp = requests.get(self.API_URL, params=params, timeout=10)
                if resp.status_code != 200:
                    print(f"Error fetching SO for {tag}: {resp.status_code} {resp.text}")
                    break
                
                data = resp.json()
                for item in data.get('items', []):
                    title = html.unescape(item.get('title', ''))
                    q_body = html.unescape(item.get('body_markdown', item.get('body', '')))
                    
                    doc_id = f"so-{item['question_id']}"
                    
                    content = f"Question: {title}\n\n{q_body}\n\n"
                    
                    # Add top answer
                    if 'answers' in item and len(item['answers']) > 0:
                        # Find accepted or highest voted
                        answers = item['answers']
                        accepted = next((a for a in answers if a.get('is_accepted')), None)
                        if not accepted:
                            accepted = answers[0]
                            
                        ans_body = html.unescape(accepted.get('body_markdown', accepted.get('body', '')))
                        content += f"Answer:\n{ans_body}"
                    
                    documents.append(Document(
                        doc_id=doc_id,
                        title=f"SO: {title}",
                        content=content,
                        source="stackoverflow",
                        library=tag,
                        doc_type="qa"
                    ))
                
                if not data.get('has_more'):
                    break
                    
                time.sleep(0.5) 
                
            except Exception as e:
                print(f"Exception fetching SO: {e}")
                break
                
        return documents
