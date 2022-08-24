from bs4 import BeautifulSoup
import requests
import re


url = ""

regex = r""

r = requests.get(url)

soup = BeautifulSoup(r.text)

links = [x.get("href") for x in soup.find_all("a")]
fixed_links = [re.sub(regex, "", l) for l in links if re.search(regex, l)]
