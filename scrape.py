from bs4 import BeautifulSoup
import requests as req

def scrape_url(url):
  res = req.get(url)
  soup = BeautifulSoup(res.content, "html.parser")
  content = [content.get_text().strip() for content in soup.find_all("p") if content.get_text().strip()] 
  return content
  
