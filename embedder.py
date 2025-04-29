import os
import json
import fitz
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from text_preprocessor import TextPreprocessor
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer


class Embedder:
    """
    Embedder with configurable multi-level chunking strategies for title, content, and details.
    chunk_method: 'sliding', 'sentence', or 'token' — applies to details level.
    chunk_params applies to details-level chunking.

    chunking_sizes: sizes for each level in characters.
      - title: first N chars of first page
      - content: first N chars of each page
      - details: sliding/sentence/token chunks of full page text
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        documents_dir: str = "documents",
        output: str = "embeddings.json",
        remove_stopwords: bool = False,
        remove_punct: bool = True,
        chunk_method: str = "sliding",
        chunk_params: dict = None,
        chunking_sizes: dict = None,
    ):
        # Preprocessor + embedder model
        self.preprocessor = TextPreprocessor(
            remove_stopwords=remove_stopwords, remove_punct=remove_punct
        )
        self.model = SentenceTransformer(model_name)

        # Chunking config for details
        self.chunk_method = chunk_method
        # defaults for details-level only
        defaults = {
            "sliding": {"size": 1000, "overlap": 500},
            "sentence": {"max_chars": 800, "overlap_sents": 1},
            "token": {"max_tokens": 256, "overlap_tokens": 64},
        }
        self.chunk_params = chunk_params or defaults[chunk_method]
        if chunk_method == "token":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Multi-level chunk sizes for title/content/details slicing
        self.chunking_sizes = chunking_sizes or {
            "title": 250,
            "content": 750,
            "details": self.chunk_params.get("size", 1000),
        }

        self.documents_dir = documents_dir
        self.output = output

    def extract_text(self, path: str) -> list[str]:
        doc = fitz.open(path)
        return [page.get_text().replace("\n", " ").strip() for page in doc]

    def chunk_text(self, text: str) -> list[str]:
        """
        Chunk full text for details level based on chunk_method & chunk_params.
        """
        m = self.chunk_method
        p = self.chunk_params

        if m == "sliding":
            step = p["overlap"]
            size = p["size"]
            return [text[i : i + size] for i in range(0, len(text), size - step)]

        if m == "sentence":
            sents = sent_tokenize(text)
            chunks, buf = [], []
            buf_chars = 0
            for sent in sents:
                if buf_chars + len(sent) > p["max_chars"]:
                    chunks.append(" ".join(buf))
                    buf = buf[-p["overlap_sents"] :]
                    buf_chars = sum(len(s) for s in buf)
                buf.append(sent)
                buf_chars += len(sent)
            if buf:
                chunks.append(" ".join(buf))
            return chunks

        if m == "token":
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            max_t, ov = p["max_tokens"], p["overlap_tokens"]
            chunks = []
            for i in range(0, len(tokens), max_t - ov):
                window = tokens[i : i + max_t]
                chunks.append(self.tokenizer.decode(window))
            return chunks

        raise ValueError(f"Unknown chunk_method: {m}")

    def embed_text(self, text: str) -> list[float]:
        cleaned = self.preprocessor.preprocess(text)
        return self.model.encode(cleaned).tolist()

    def process_document(self, path: str) -> list[dict]:
        """
        Extract and embed at three levels: title, content, and details.
        Returns a list of entries, each with nested embeddings.
        """
        pages = self.extract_text(path)
        entries = []

        # Title level: use first page and truncate
        title_text = pages[0][: self.chunking_sizes["title"]]
        title_emb = self.embed_text(title_text)

        for page_text in pages:
            # Content level: truncate each page
            content_text = page_text[: self.chunking_sizes["content"]]
            content_emb = self.embed_text(content_text)

            # Details level: full page chunks
            detail_chunks = self.chunk_text(page_text)
            for chunk in detail_chunks:
                detail_emb = self.embed_text(chunk)

                entries.append(
                    {
                        "title": {"text": title_text, "embedding": title_emb},
                        "content": {"text": content_text, "embedding": content_emb},
                        "details": {"text": chunk, "embedding": detail_emb},
                    }
                )

        return entries

    def process_all_documents(self):
        all_e = []
        for fname in tqdm(os.listdir(self.documents_dir), desc="Processing Documents"):
            if not fname.lower().endswith((".pdf", ".txt")):
                continue
            path = os.path.join(self.documents_dir, fname)
            all_e.extend(self.process_document(path))

        with open(self.output, "w", encoding="utf-8") as f:
            json.dump(all_e, f, indent=2)

        print(f"Generated {len(all_e)} entries into {self.output}")


if __name__ == "__main__":
    e = Embedder(
        chunk_method="sliding",
        chunk_params={"size": 1000, "overlap": 500},
        chunking_sizes={"title": 250, "content": 750, "details": 1000},
    )
    e.process_all_documents()
