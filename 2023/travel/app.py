import airportsdata
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from typing import AbstractSet


def raise_for_status(response: requests.Response):
    if not response.ok:
        st.warning(response.text, icon="💥")
        response.raise_for_status()


@st.cache_data
def airports() -> pd.DataFrame:
    """
    https://pbpython.com/pandas-html-table.html
    https://medium.com/analytics-vidhya/web-scraping-a-wikipedia-table-into-a-dataframe-c52617e1f451
    """
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_airports_in_the_United_States", match="SEA")
    assert len(tables) == 1, "Table should be unique"
    return tables[0]


@st.cache_data
def seattle_destinations() -> set[str]:
    response = requests.get("https://www.seattle-airport.com/seatac-departures")
    raise_for_status(response)
    soup = BeautifulSoup(response.text, 'html.parser')
    # https://stackoverflow.com/a/45647689/
    divs = soup.find_all("div", {'class':"flight-col flight-col__dest"})
    return set([
        div.text.split("(")[-1].strip(" )")
        for div in divs
        if "(" in div.text
    ])



@st.cache_data
def destinations(url: str, class_: str) -> AbstractSet[str]:
    response = requests.get(url)
    raise_for_status(response)
    # st.code(response.text, language="html")
    soup = BeautifulSoup(response.text, 'html.parser')
    # https://stackoverflow.com/a/45647689/
    divs = soup.find_all("div", {'class':class_})
    # st.write(f"Found {len(divs)} divs from {url}")
    # for div in divs:
    #     st.write("----")
    #     st.write(str(div.text))
    #     st.write("----")
    return set([
        div.text.split("(")[-1].strip(" )")
        for div in divs
        if "(" in div.text
    ])





@st.cache_data
def flights_from(airport: str, date):
    url = "http://api.aviationstack.com/v1/flights"
    params ={
        # "flight_date": str(date), 
        "dep_iata": airport, 
        **st.secrets.aviationstack}
    response = requests.get(url, params=params)
    if not response.ok:
        st.warning(response.text, icon="💥")
        response.raise_for_status()
    return response.json()



@st.cache_data
def aggregated_weather(month: int, latitude, longitude):
    # Their historical data isn't in the free tier 😡.
    url = "http://history.openweathermap.org/data/2.5/aggregated/month"
    params = {
        "lat": latitude,
        "lon": longitude,
        "month": month,
        **st.secrets.openweathermap
    }
    response = requests.get(url, params=params)
    raise_for_status(response)
    return response.json()


@st.cache_data
def airports_data() -> pd.DataFrame:
    loaded = airportsdata.load("IATA")
    return pd.DataFrame(loaded.values())


st.set_page_config("Travel", page_icon=":airplane:")

st.title("Travel")







airports_wiki = airports()

st.write(airports_wiki.head())

selected_airports = st.multiselect("Origins", airports_wiki["Airport"].dropna(), default=["Seattle–Tacoma International Airport", "Chicago O'Hare International Airport"])

# https://stackoverflow.com/a/12065904
selected_rows = airports_wiki[airports_wiki["Airport"].isin(selected_airports)]


date = st.date_input("Date", value="today")

airports_lib = airports_data()





ohare_dest = destinations("https://www.airport-ohare.com/departures.php", "flight-col flight-col__dest-term")
seat_dest = destinations("https://www.seattle-airport.com/seatac-departures", "flight-col flight-col__dest")
dest = ohare_dest.intersection(seat_dest)





us_destinations = airports_lib[(airports_lib["country"] == "US") & airports_lib["iata"].isin(dest)]

st.write(us_destinations.head())


st.map(pd.DataFrame(us_destinations))

month = st.number_input("Month", 1, 12, 2)

# st.write(us_destinations.apply(lambda row: aggregated_weather(month, row["lat"], row["lon"]), axis=1).head())


# TODO: https://stackoverflow.com/questions/9751197/opening-pdf-urls-with-pypdf
# TODO: https://www.flychicago.com/SiteCollectionDocuments/O%27Hare/MyFlight/DomesticNonstops.pdf
# https://pypi.org/project/pypdf/
# https://stackoverflow.com/a/36066208/2543689
# https://pypi.org/project/fake-useragent/
# https://github.com/dkmiller/tidbits/blob/master/2023/kubernetes/examples/e2e/images/ui/ui/probe.py
# from urllib2 import Request, urlopen
# from pypdf import PdfReader
# from io import BytesIO
# # Cloudflare blocks this, either via requests or via
# # https://github.com/pyppeteer/pyppeteer
# url = "https://www.flychicago.com/SiteCollectionDocuments/O%27Hare/MyFlight/DomesticNonstops.pdf"
# writer = PdfFileWriter()




# remoteFile = urlopen(Request(url)).read()
# content = requests.get(url).content
# st.write(f"Got {len(content)} bytes")
# memoryFile = BytesIO(content)

# # st.write(memoryFile)

# pdfFile = PdfReader(memoryFile)

# st.write(pdfFile)


st.stop()

text = requests.get("https://www.seattle-airport.com/seatac-departures").text

st.write(len(text))

from bs4 import BeautifulSoup # library to parse HTML documents

soup = BeautifulSoup(text, 'html.parser')
st.write(str(soup)[:100])
alls = soup.find_all("div", {'class':"flight-col flight-col__dest"})
st.write(len(alls))
for a in alls:
    st.write("----")
    st.write(a.text)
    st.write(a.text.split("(")[-1].strip(" )"))
    st.write("----")
# indiatable=soup.find('div',{'class':"flight-row"})
# st.write(indiatable is None)
# st.write(indiatable)
# __df = pd.read_html("https://www.seattle-airport.com/seatac-departures")
# st.write(len(__df))

st.stop()

x = selected_rows["IATA"].map(lambda iata: flights_from(iata, date))

# st.write(type(x))

st.write("Got flights!")

for y in reversed(x):
    st.write(y)
    st.write(y["pagination"])
    st.write(y["data"][0])


# response = requests.get("http://api.aviationstack.com/v1/cities?access_key=")
# st.write(response.json())

# https://api.aviationstack.com/v1/flights
#     ? access_key = YOUR_ACCESS_KEY
#     & flight_date = 2019-12-31
# &dep_iata=