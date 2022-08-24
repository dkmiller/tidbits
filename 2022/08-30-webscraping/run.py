from bs4 import BeautifulSoup
import requests
import re


url = ""
str = r"\-BLUR"

r = requests.get(url)

soup = BeautifulSoup(r.text)

links = [x.get("href") for x in soup.find_all("a")]
