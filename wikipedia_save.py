from wikipedia_fetcher import WikipediaFetcher

wf = WikipediaFetcher(user_agent="WikiBot/2.0")
title = "Cat"

# 1) Fetch the raw extract
text = wf.fetch_text(title)

# 2) Write it to a .txt file (spaces in title are underscores)
filename = f"{title}.txt"
with open(filename, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Saved raw text to {filename}")
