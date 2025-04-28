import os
import json
import fitz
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

'''
This program will extract the data from the pdf break it down into 3 parts:
- Title Summary (very low dimensional embedding for rough content outline)
- Content Summary (medium dimensional embedding for paragraph content outlines)
- Fine Summary (high dimensional embedding for detailed outline)

this will act as a sieve for the input embedding to find the best content that answers the query.

we need to add this data to a data file that the search engine can reference
'''

class DocExtractor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents_dir = 'documents'
        self.output = 'embeddings.json'

        self.chunking_sizes = {
            'title': 250,
            'content': 750,
            'details': 1000

        }

    def extract_text(self, file_path):
        doc = fitz.open(file_path)
        return [{'page_num': i+1, 'text': page.get_text().replace('\n', ' ').strip()} for i, page in enumerate(doc)]

    def chunk_text(self, text, chunk_size):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size // 2)]

    def embed_text(self, text):
        return self.model.encode(text).tolist()

    def process_document(self, file_path):
        pages = self.extract_text(file_path)
        entries = []

        title_text = pages[0]['text'][:self.chunking_sizes['title']]
        title_embedding = self.embed_text(title_text)

        for page in pages:
            content_text = page['text'][:self.chunking_sizes['content']]
            content_embedding = self.embed_text(content_text)

            detail_chunks = self.chunk_text(page['text'], self.chunking_sizes['details'])
            for idx, chunk in enumerate(detail_chunks):
                detail_embedding = self.embed_text(chunk)

                entries.append({
                    'title': {
                        'text': title_text,
                        'embedding': title_embedding,
                    },
                    'content': {
                        'text': content_text,
                        'embedding': content_embedding,
                    },
                    'details': {
                        'text': chunk,
                        'embedding': detail_embedding,
                    },
                })

        return entries

    def process_all_documents(self):
        all_embeddings = []

        pdf_files = [f for f in os.listdir(self.documents_dir) 
                    if f.lower().endswith(".pdf")]

        for filename in tqdm(pdf_files, desc="Processing Documents"):
            file_path = os.path.join(self.documents_dir, 'Cat.txt')
            all_embeddings.extend(self.process_document(file_path))

        # Save embeddings with reference keys
        with open(self.output, 'w') as f:
            json.dump(all_embeddings, f, indent=2)

        print(f"Generated {len(all_embeddings)} embedding entries")

if __name__ == '__main__':
    DocExtractor().process_all_documents()


