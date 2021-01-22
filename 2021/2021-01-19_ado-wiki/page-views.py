from typing import Dict, List
from azure.identity import DefaultAzureCredential
import hydra
import jmespath
import logging
import pandas as pd
import requests


log = logging.getLogger(__name__)


def get_auth_headers(*scopes: List[str]) -> Dict[str, str]:
    credential = DefaultAzureCredential()
    access_token = credential.get_token(*scopes).token
    headers = {"Authorization": f"Bearer {access_token}"}
    return headers


def get_wiki_id(base_url: str, wiki_name: str, headers: Dict[str, str]) -> str:
    """
    Not necessary, but good use of JMESPath.
    """
    url = f"{base_url}/_apis/wiki/wikis"
    r = requests.get(url, headers=headers)
    wikis = r.json()
    wiki_id = jmespath.search(f"value[?name == '{wiki_name}'].id", wikis)[0]

    return wiki_id


def get_pages_batch(
    base_url: str, wiki_name: str, json: dict, headers: Dict[str, str]
) -> requests.Response:
    url = (
        f"{base_url}/_apis/wiki/wikis/{wiki_name}/pagesbatch?api-version=6.0-preview.1"
    )
    log.info(f"Calling POST {url} with body {json}")
    rv = requests.post(url, headers=headers, json=json)
    return rv


def get_all_page_stats(
    base_url: str, wiki_name: str, headers: Dict[str, str], top: int = 100
) -> pd.DataFrame:

    count = None
    continuation_token = None
    body = {"top": top, "pageViewsForDays": 30}
    pages = []

    while count != 0:
        if continuation_token is not None:
            body["continuationToken"] = continuation_token

        r = get_pages_batch(base_url, wiki_name, body, headers)
        json = r.json()
        new_pages = json["value"]
        pages.extend(new_pages)
        count = json["count"]
        continuation_token = r.headers.get("x-ms-continuationtoken")
        log.info(f"Got {len(new_pages)} pages.")

    d = {"day": [], "path": [], "count": []}

    for page in pages:
        path = page["path"]
        if "viewStats" in page:
            for stat in page["viewStats"]:
                d["day"].append(stat["day"])
                d["path"].append(path)
                d["count"].append(stat["count"])
        else:
            log.warning(f"Page {path} does not have view stats.")

    rv = pd.DataFrame(d)
    return rv


def summarize_wiki_stats(stats: pd.DataFrame, top_pages: int) -> None:
    log.info(f"Page view stats: {stats}")

    log.info(f"Got page view stats for {stats.path.nunique()} pages")
    total_views = stats["count"].sum()
    log.info(f"Total page views: {total_views}")

    agg_stats = (
        stats.groupby(["path"])["count"]
        .agg("sum")
        .sort_values(ascending=False)
        .head(top_pages)
    )
    log.info(f"Aggregated stats: {agg_stats.to_string()}")


@hydra.main(config_name="config")
def main(config):
    log.info(f"Configuration: {config}")

    headers = get_auth_headers(*config.scopes)

    wiki_id = get_wiki_id(config.url, config.wiki, headers=headers)
    log.info(f"ID of wiki {config.wiki} = {wiki_id}")

    stats = get_all_page_stats(config.url, config.wiki, headers)
    summarize_wiki_stats(stats, config.top_pages)


if __name__ == "__main__":
    main()
