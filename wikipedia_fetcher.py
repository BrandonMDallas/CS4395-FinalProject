import requests

def fetch_wikipedia_text_via_api(title: str) -> str:
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    resp = requests.get(endpoint, params=params, headers={"User-Agent": "MyBot/1.0"})
    resp.raise_for_status()
    data = resp.json()["query"]["pages"]
    page = next(iter(data.values()))
    text = page.get("extract", "")
    return text

if __name__ == "__main__":
    raw = fetch_wikipedia_text_via_api("Natural_language_processing")
    print("raw length:", len(raw))
    print(raw[:500])