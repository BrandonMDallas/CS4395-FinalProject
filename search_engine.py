import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import faiss
import json
from sklearn.preprocessing import normalize
from typing import List, Union, Optional
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder

from text_preprocessor import TextPreprocessor


class SearchEngine:
    """
    A hybrid search engine combining BM25, FAISS semantic search, and
    cross-encoder reranking, using TextPreprocessor for all text cleaning.

    Args:
        embedding_model: model name or SentenceTransformer instance
        cross_encoder: model name or CrossEncoder instance
        index_preprocessor: TextPreprocessor to use when indexing (defaults to removing stopwords)
        query_preprocessor: TextPreprocessor to use for queries (defaults to keeping stopwords)
    """

    def __init__(
        self,
        embeddings_path: str = "embeddings.json",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        # --- 1) Load embeddings.json into a DataFrame
        with open(embeddings_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        self.df = pd.DataFrame(entries)

        # --- 2) Build FAISS index over the "details" embeddings
        detail_vecs = np.array(self.df["details"].tolist(), dtype="float32")
        faiss.normalize_L2(detail_vecs)

        dim = detail_vecs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # inner-product for cosine
        self.faiss_index.add(detail_vecs)

        # --- 3) Load the same SentenceTransformer for query encoding
        self.embedder = SentenceTransformer(model_name)

    def index_data(self, documents: List[str]) -> None:
        """
        Split raw documents into true sentences, preprocess each one,
        then build BM25 & FAISS indices over the cleaned text.
        """
        # 1) Sentence‐split the *raw* docs, then clean each sentence
        sentences = []
        for doc in documents:
            raw_sents = sent_tokenize(doc)
            for sent in raw_sents:
                cleaned = self.index_preprocessor.preprocess(sent)
                if cleaned:
                    sentences.append((sent, cleaned))

        # 2) Build DataFrame with BOTH original and processed
        self.df = pd.DataFrame(sentences, columns=["original_text", "processed_text"])

        # 3) BM25 lexical index over the processed tokens
        tokenized = [txt.split() for txt in self.df["processed_text"]]
        self.bm25 = BM25Okapi(tokenized)

        # 4) FAISS semantic index over the same embeddings
        embeddings = self.embedding_model.encode(
            self.df["processed_text"].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = normalize(embeddings, axis=1)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype(np.float32))

    def semantic_search(self, query: str, top_k: int = 5):
        """
        Returns the top_k text chunks most semantically similar to `query`.
        Each result is a dict with:
          - 'text':   the raw snippet
          - 'score':  cosine similarity (0–1)
        """
        # encode & normalize
        q_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        # search
        distances, indices = self.faiss_index.search(q_vec, top_k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            snippet = self.df.iloc[idx]["text"]
            results.append({"text": snippet, "score": float(score)})
        return results

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Hybrid search:
        1) BM25 scoring (lexical)
        2) FAISS semantic retrieval
        3) weighted score fusion
        4) cross-encoder reranking
        """
        # 1) Preprocess the query (keeps stopwords as configured)
        proc_q = self.query_preprocessor.preprocess(query)
        tokens = proc_q.split()

        # 2) BM25 scores → normalize to [0,1]
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_scores = normalize(bm25_scores.reshape(1, -1))[0]

        # 3) FAISS semantic retrieval (may return -1 for “empty” slots)
        q_emb = self.embedding_model.encode([proc_q])
        q_emb = normalize(q_emb, axis=1).astype(np.float32)
        faiss_scores, faiss_idxs = self.faiss_index.search(q_emb, top_k * 3)
        faiss_scores = faiss_scores[0]
        raw_idxs = faiss_idxs[0]

        # 4) Fuse scores, but skip any negative indices
        fused = {}
        for idx, sem_score in zip(raw_idxs, faiss_scores):
            if idx < 0:
                continue
            fused[idx] = 0.6 * sem_score + 0.4 * bm25_scores[idx]

        # 5) Take the top-2*top_k candidates for reranking
        top_initial = sorted(fused, key=fused.get, reverse=True)[: top_k * 2]

        # 6) Gather their original text via integer‐position indexing
        candidates = [self.df.iloc[i]["original_text"] for i in top_initial]

        # 7) Rerank with cross-encoder
        pairs = [[query, cand] for cand in candidates]
        rerank_scores = self.cross_encoder.predict(pairs)

        # 8) Pick final top_k by rerank score
        order = np.argsort(rerank_scores)[::-1][:top_k]
        final_idxs = [top_initial[i] for i in order]

        # 9) Return those rows, reset the index
        return self.df.iloc[final_idxs].reset_index(drop=True)


if __name__ == "__main__":
    se = SearchEngine()
    hits = se.semantic_search("Do cats have memory?", top_k=3)
    for res in hits:
        print(f"Score: {res['score']:.4f}\nSnippet: {res['text']}\n---\n")
