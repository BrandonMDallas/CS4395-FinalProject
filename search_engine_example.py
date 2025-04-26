from search_engine import SearchEngine
from wikipedia_fetcher import WikipediaFetcher


def main():
    # 1) Define which pages to fetch; note underscores instead of spaces
    topics = [
        "Python_(programming_language)",
        "Racket_(programming_language)",
        "Lisp_(programming_language)",
        "Functional_programming",
        "Lambda_calculus",
    ]

    # 2) Fetch the raw extracts from Wikipedia
    print("Status: Fetching Wikipedia pages...")
    wf = WikipediaFetcher(user_agent="SearchEngineExample/1.0")
    docs = wf.fetch_bulk(topics)
    print(f"Status: Fetched {len(docs)} pages.")

    # 3) Initialize and build the search index
    print("Status: Indexing documents...")
    engine = SearchEngine()
    engine.index_data(docs)
    print("Status: Indexing complete.")

    # 4) Run a query
    query = "What is python?"
    print(f"Status: Running search for query: '{query}'...")
    results = engine.search(query, top_k=5)
    print("Status: Search complete. Displaying results:\n")

    # 5) Display the top-5 matching sentences
    print(f"Search Results for: “{query}”\n" + "-" * 50)
    for rank, row in enumerate(results.itertuples(), start=1):
        print(f"{rank}. {row.original_text}\n")


if __name__ == "__main__":
    main()
