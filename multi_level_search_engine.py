import json
import ijson
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import faiss
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, CrossEncoder
from text_preprocessor import TextPreprocessor
from tqdm import tqdm
import os


class MultiLevelSearchEngine:
    """
    Hybrid search engine over multi-level embeddings (title, content, details),
    building the FAISS index by streaming to avoid OOM/segfaults, with a
    byte-offset progress bar.
    """

    def __init__(
        self,
        embeddings_path: str = "embeddings.json",
        model_name: str = "all-MiniLM-L6-v2",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        weights: dict = None,
    ):
        # --- load the multi‐level entries for DataFrame/BM25
        with open(embeddings_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        df = pd.json_normalize(entries)
        df = df.rename(
            columns={
                "title.text": "title_text",
                "content.text": "content_text",
                "details.text": "details_text",
            }
        )
        self.df = df

        # default weights if not provided
        w = weights or {"title": 0.2, "content": 0.3, "details": 0.5}
        self.weights = w

        # --- build FAISS index by streaming with byte-offset progress
        # 1) peek dimension from the first entry
        with open(embeddings_path, "rb") as f:
            parser = ijson.items(f, "item")
            first = next(parser)
            dim = len(first["title"]["embedding"])

        # 2) prepare index and progress bar
        file_size = os.path.getsize(embeddings_path)
        idx = faiss.IndexFlatIP(dim)

        with open(embeddings_path, "rb") as f, tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Indexing JSON"
        ) as pbar:

            parser = ijson.items(f, "item")
            last_pos = f.tell()

            for entry in parser:
                # build the weighted vector
                t = np.array(entry["title"]["embedding"], dtype="float32")
                c = np.array(entry["content"]["embedding"], dtype="float32")
                d = np.array(entry["details"]["embedding"], dtype="float32")

                vec = w["title"] * t + w["content"] * c + w["details"] * d
                # normalize and add to FAISS
                faiss.normalize_L2(vec.reshape(1, -1))
                idx.add(vec.reshape(1, -1))

                # update progress by bytes consumed
                new_pos = f.tell()
                pbar.update(new_pos - last_pos)
                last_pos = new_pos

        self.faiss_index = idx

        # --- BM25 over combined raw text
        raw_texts = (
            df["title_text"] + " " + df["content_text"] + " " + df["details_text"]
        )
        self.index_preprocessor = TextPreprocessor(remove_stopwords=True)
        tokenized = [
            self.index_preprocessor.preprocess(txt).split() for txt in raw_texts
        ]
        self.bm25 = BM25Okapi(tokenized)

        # --- models for querying
        self.query_preprocessor = TextPreprocessor(remove_stopwords=False)
        self.embedder = SentenceTransformer(model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_name)

        # keep for later reference
        self.df["combined_text"] = raw_texts
        self.df["bm25_tokens"] = tokenized

    def lexical_search(self, query: str, top_k: int = 10):
        """
        Pure-lexical (BM25) search over combined_text.
        """
        # 1) preprocess & tokenize
        proc_q = self.query_preprocessor.preprocess(query)
        tokens = proc_q.split()

        # 2) score with BM25
        bm25_scores = self.bm25.get_scores(tokens)

        # 3) pick top_k indices
        top_idxs = np.argsort(bm25_scores)[::-1][:top_k]

        # 4) assemble results
        results = []
        for i in top_idxs:
            row = self.df.iloc[i]
            results.append(
                {
                    "score": float(bm25_scores[i]),
                    "title": row["title_text"],
                    "content": row["content_text"],
                    "details": row["details_text"],
                }
            )
        return results

    def semantic_search(self, query: str, top_k: int = 5):
        proc = self.query_preprocessor.preprocess(query)
        q_vec = self.embedder.encode([proc], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        distances, indices = self.faiss_index.search(q_vec, top_k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            row = self.df.iloc[idx]
            results.append(
                {
                    "score": float(score),
                    "title": row["title_text"],
                    "content": row["content_text"],
                    "details": row["details_text"],
                }
            )
        return results

    def hybrid_search(self, query: str, top_k: int = 10):
        """
        1) BM25 lexical scoring
        2) FAISS semantic retrieval
        3) fuse with configured blend (lex weight 0.4 / sem weight 0.6)
        4) cross‐encoder rerank top candidates
        """
        # lexical
        proc_q = self.query_preprocessor.preprocess(query)
        tokens = proc_q.split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_norm = normalize(bm25_scores.reshape(1, -1))[0]

        # semantic
        q_emb = self.embedder.encode([proc_q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        sem_scores, sem_idxs = self.faiss_index.search(q_emb, top_k * 3)
        sem_scores, sem_idxs = sem_scores[0], sem_idxs[0]

        # fuse
        fused = {}
        for score, i in zip(sem_scores, sem_idxs):
            if i < 0:
                continue
            fused[i] = 0.6 * score + 0.4 * bm25_norm[i]

        # pick top initial
        top_init = sorted(fused, key=fused.get, reverse=True)[: top_k * 2]
        candidates = [self.df.iloc[i]["combined_text"] for i in top_init]

        # rerank
        pairs = [[query, c] for c in candidates]
        rerank_scores = self.cross_encoder.predict(pairs)
        order = np.argsort(rerank_scores)[::-1][:top_k]
        final_idxs = [top_init[i] for i in order]

        return self.df.iloc[final_idxs].reset_index(drop=True)

    def hybrid_search_dicts(self, query: str, top_k: int = 5, alpha: float = 0.5):
        proc_q = self.query_preprocessor.preprocess(query)
        tokens = proc_q.split()

        bm25_scores = self.bm25.get_scores(tokens)
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-8)

        q_vec = self.embedder.encode([proc_q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)
        dists, _ = self.faiss_index.search(q_vec, self.df.shape[0])
        sem_scores = dists[0]
        sem_min, sem_max = sem_scores.min(), sem_scores.max()
        sem_norm = (sem_scores - sem_min) / (sem_max - sem_min + 1e-8)

        fused = alpha * bm25_norm + (1 - alpha) * sem_norm
        top_idxs = np.argsort(fused)[::-1][:top_k]

        # Build dicts
        return [
            {
                "score": float(fused[i]),
                "title": self.df.at[i, "title_text"],
                "content": self.df.at[i, "content_text"],
                "details": self.df.at[i, "details_text"],
            }
            for i in top_idxs
        ]


if __name__ == "__main__":
    se = MultiLevelSearchEngine(
        embeddings_path="embeddings.json",
        weights={"title": 0.1, "content": 0.3, "details": 0.6},
    )

    query = "Where did the word cat come from?"
    top_k = 5

    for name, search_fn in [
        ("Lexical", se.lexical_search),
        ("Semantic", se.semantic_search),
        ("Hybrid", se.hybrid_search_dicts),
    ]:
        print(f"{name} top-{top_k}:")
        for r in search_fn(query, top_k=top_k):
            print(f"Score {r['score']:.3f}")
            detail = r.get("details") or r.get("content") or ""
            print(" Details:", detail)
            print("---")
        print()
