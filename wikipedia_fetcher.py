import requests
from typing import List


class WikipediaFetcher:
    """
    Wrapper around the Wikipedia API for fetching page extracts.

    Usage:
      wf = WikipediaFetcher(user_agent="MyCustomBot/1.0")
      single_text = wf.fetch_text("Python_(programming_language)")
      multiple_texts = wf.fetch_bulk(["Python_(programming_language)", "Racket_(programming_language)"])
    """

    def __init__(
        self,
        endpoint: str = "https://en.wikipedia.org/w/api.php",
        user_agent: str = "MyBot/1.0",
    ):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def fetch_text(self, title: str) -> str:
        """
        Fetch the plain-text extract for a single page.

        Args:
            title: The exact Wikipedia page title (spaces replaced with underscores).

        Returns:
            The page extract as a string (empty if page not found).
        """
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "titles": title,
            "format": "json",
        }
        resp = self.session.get(self.endpoint, params=params)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        return page.get("extract", "")

    def fetch_bulk(self, titles: List[str]) -> List[str]:
        """
        Fetch extracts for multiple pages in sequence.

        Args:
            titles: A list of Wikipedia page titles.

        Returns:
            A list of page extracts in the same order as the titles.
        """
        return [self.fetch_text(title) for title in titles]


if __name__ == "__main__":
    wf = WikipediaFetcher(user_agent="WikiBot/2.0")
    print(wf.fetch_text("Natural_language_processing")[:500])
