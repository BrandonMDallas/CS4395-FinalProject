import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import faiss
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize
from typing import List, Tuple

# ----------------------
# 1. Preprocessing & Embeddings
# ----------------------
import spacy, spacy.cli

try:
    spacy.require_gpu()  # optional, if you want GPU
    nlp = spacy.load("en_core_web_trf", disable=["parser", "ner"])
except OSError:
    # model not found → download it, then load
    spacy.cli.download("en_core_web_trf")
    nlp = spacy.load("en_core_web_trf", disable=["parser", "ner"])

def preprocess_text(text: str) -> str:
    """Advanced text cleaning with lemmatization and entity awareness"""
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight but powerful
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Reranker

# ----------------------
# 2. Indexing Pipeline
# ----------------------
class SearchEngine:
    def __init__(self):
        self.df = None
        self.bm25 = None
        self.faiss_index = None
        self.embedding_model = embedding_model
    
    def index_data(self, documents: List[str]):
        """Process and index documents"""
        # Preprocess
        self.df = pd.DataFrame({"raw_text": documents})
        self.df["processed_text"] = self.df["raw_text"].apply(preprocess_text)
        
        # BM25 Index
        tokenized_docs = [doc.split() for doc in self.df["processed_text"]]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Vector Embeddings
        embeddings = self.embedding_model.encode(self.df["processed_text"], show_progress_bar=True)
        embeddings = normalize(embeddings, axis=1)  # Crucial for cosine similarity
        
        # FAISS Index
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings.astype(np.float32))
        
    def search(self, query: str, top_k: int = 10, rerank: bool = True) -> pd.DataFrame:
        """Hybrid search with optional reranking"""
        # Preprocess query
        processed_query = preprocess_text(query)
        
        # BM25 Search (lexical)
        tokenized_query = processed_query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Vector Search (semantic)
        query_embedding = self.embedding_model.encode([processed_query])
        query_embedding = normalize(query_embedding, axis=1)
        faiss_scores, faiss_indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        # Combine results
        bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        combined_indices = np.union1d(faiss_indices[0], bm25_indices)
        
        # Rerank with Cross-Encoder
        if rerank:
            candidates = self.df.iloc[combined_indices]["raw_text"].tolist()
            pairs = [[query, candidate] for candidate in candidates]
            rerank_scores = cross_encoder.predict(pairs)
            sorted_indices = np.argsort(rerank_scores)[::-1]
            final_indices = combined_indices[sorted_indices][:top_k]
        else:
            final_indices = combined_indices[:top_k]
            
        return self.df.iloc[final_indices]

# ----------------------
# 3. Usage Example
# ----------------------
if __name__ == "__main__":
    # Sample documents (replace with your scraped data)
    documents = [
        "JavaScript type errors occur when you try to access undefined variables.",
        "Python lists are mutable ordered sequences of elements.",
        "TypeScript adds static typing to JavaScript to catch errors early.",
        "HTTP 404 errors indicate missing resources on the server.",
        "React useState hook manages component state in functional components."
    ]
    
    # Build search engine
    engine = SearchEngine()
    engine.index_data(documents)
    
    # Execute query
    query = "what animals make good pets"
    results = engine.search(query, top_k=2)
    
    print("\nSearch Results:")
    print(results[["raw_text"]])