from sentence_transformers import CrossEncoder, SentenceTransformer
from preprocessing import preprocess_text  # Ensure this keeps numbers
from rank_bm25 import BM25Okapi
from scrape import scrape_url
import pandas as pd
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize

# Initialize models once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

class SearchEngine:
    def __init__(self):
        self.df = pd.DataFrame()
        self.bm25 = None
        self.faiss_index = None
        self.embedding_model = embedding_model
        
    def index_data(self, documents: List[str]):
        """Process and index documents while preserving key information"""
        # Use proper sentence tokenization
        sentences = []
        for doc in documents:
            # Clean but preserve numerical facts
            cleaned = preprocess_text(doc)  
            sentences.extend(sent_tokenize(cleaned))
            
        self.df = pd.DataFrame({"original_text": sentences})
        self.df['processed_text'] = self.df['original_text'].apply(
            lambda x: preprocess_text(x)
        )
        
        # BM25 Index
        tokenized_docs = [doc.split() for doc in self.df['processed_text']]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # FAISS Index
        embeddings = self.embedding_model.encode(
            self.df['processed_text'].tolist(), 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        embeddings = normalize(embeddings, axis=1)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
      """Hybrid search with normalized score fusion"""
      # Process query without summarization
      processed_query = preprocess_text(query, remove_stopwords=False)  # Keep stopwords for better matching
      
      # BM25 Retrieval
      tokenized_query = processed_query.split()
      bm25_scores = self.bm25.get_scores(tokenized_query)
      bm25_scores = normalize(bm25_scores.reshape(1, -1))[0]  # Normalize to [0,1]
      
      # Dense Retrieval
      query_embedding = self.embedding_model.encode([processed_query])
      query_embedding = normalize(query_embedding, axis=1)
      faiss_scores, faiss_indices = self.faiss_index.search(
          query_embedding.astype(np.float32), 
          top_k * 3  # Wider initial pool
      )
      faiss_scores = faiss_scores[0]
      
      # Combine scores using weighted sum
      combined_scores = {}
      for idx, score in zip(faiss_indices[0], faiss_scores):
          combined_scores[idx] = 0.6 * score + 0.4 * bm25_scores[idx]
          
      # Rerank top candidates with cross-encoder
      sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k * 2]
      candidates = self.df.iloc[sorted_indices]['original_text'].tolist()
      
      pairs = [[query, cand] for cand in candidates]
      rerank_scores = cross_encoder.predict(pairs)
      
      # Fix: Convert NumPy array indexing to list comprehension
      sorted_by_rerank = np.argsort(rerank_scores)[::-1][:top_k]
      final_indices = [sorted_indices[idx] for idx in sorted_by_rerank]
      
      return self.df.iloc[final_indices]

if __name__ == '__main__':
    documents = scrape_url('https://stackoverflow.com/questions/54649465/how-to-do-try-catch-and-finally-statements-in-typescript')
    
    engine = SearchEngine()
    engine.index_data(documents)
    
    query = "with typescript you can set unknown as what"
    results = engine.search(query, top_k=2)
    print('\nSearch Results:')
    print(results[['original_text']])