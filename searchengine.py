import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import faiss
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
        embedding_model: Optional[Union[str, SentenceTransformer]] = "all-MiniLM-L6-v2",
        cross_encoder: Optional[
            Union[str, CrossEncoder]
        ] = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        index_preprocessor: Optional[TextPreprocessor] = None,
        query_preprocessor: Optional[TextPreprocessor] = None,
    ):
        # load or accept embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = embedding_model

        # load or accept cross-encoder
        if isinstance(cross_encoder, str):
            self.cross_encoder = CrossEncoder(cross_encoder)
        else:
            self.cross_encoder = cross_encoder

        # text preprocessors
        # default index_preprocessor: remove stopwords/punct
        self.index_preprocessor = index_preprocessor or TextPreprocessor()
        # default query_preprocessor: keep stopwords (better BM25 matching)
        self.query_preprocessor = query_preprocessor or TextPreprocessor(
            remove_stopwords=False
        )

        # placeholders for data & indices
        self.df = pd.DataFrame()
        self.bm25 = None
        self.faiss_index = None

    def index_data(self, documents: List[str]) -> None:
        """
        Split raw documents into sentences, preprocess, and build BM25 & FAISS indices.
        """
        # 1) clean + sentence-split
        sentences = []
        for doc in documents:
            cleaned = self.index_preprocessor.preprocess(doc)
            sentences.extend(sent_tokenize(cleaned))

        # 2) build DataFrame
        self.df = pd.DataFrame({"original_text": sentences})
        self.df["processed_text"] = self.df["original_text"].apply(
            self.index_preprocessor.preprocess
        )

        # 3) BM25 lexical index
        tokenized = [text.split() for text in self.df["processed_text"]]
        self.bm25 = BM25Okapi(tokenized)

        # 4) FAISS semantic index
        embeddings = self.embedding_model.encode(
            self.df["processed_text"].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = normalize(embeddings, axis=1)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Hybrid search:
          1) BM25 scoring (lexical)
          2) FAISS semantic retrieval
          3) weighted score fusion
          4) cross-encoder reranking
        """
        # preprocess query (keep stopwords)
        proc_q = self.query_preprocessor.preprocess(query)

        # BM25 scores → normalize [0,1]
        tokens = proc_q.split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_scores = normalize(bm25_scores.reshape(1, -1))[0]

        # FAISS dense retrieval
        q_emb = self.embedding_model.encode([proc_q])
        q_emb = normalize(q_emb, axis=1).astype(np.float32)
        faiss_scores, faiss_idxs = self.faiss_index.search(q_emb, top_k * 3)
        faiss_scores = faiss_scores[0]

        # fuse scores: 60% semantic + 40% lexical
        fused = {
            idx: 0.6 * sem + 0.4 * bm25_scores[idx]
            for idx, sem in zip(faiss_idxs[0], faiss_scores)
        }

        # pick top candidates for rerank
        top_initial = sorted(fused, key=fused.get, reverse=True)[: top_k * 2]
        candidates = self.df.loc[top_initial, "original_text"].tolist()
        pairs = [[query, c] for c in candidates]
        rerank_scores = self.cross_encoder.predict(pairs)

        # final top_k by rerank score
        order = np.argsort(rerank_scores)[::-1][:top_k]
        final_idxs = [top_initial[i] for i in order]

        return self.df.loc[final_idxs].reset_index(drop=True)
