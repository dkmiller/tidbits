from typing import Dict, List
from azure.identity import DefaultAzureCredential
import hydra
import jmespath
import logging
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


def get_all_page_stats(
    base_url: str, wiki_name: str, headers: Dict[str, str], top: int = 100
):
    url = (
        f"{base_url}/_apis/wiki/wikis/{wiki_name}/pagesbatch?api-version=6.0-preview.1"
    )
    body = {"top": top, "pageViewsForDays": 30}
    r = requests.post(url, headers=headers, json=body)
    json = r.json()

    print(json)
    # TODO: handle pagination.

    pages = json["value"]
    for page in pages:
        path = page["path"]
        log.info(path)
        for stat in page["viewStats"]:
            print(stat)
        print(page)


@hydra.main(config_name="config")
def main(config):
    log.info(f"Configuration: {config}")

    headers = get_auth_headers(*config.scopes)

    wiki_id = get_wiki_id(config.url, config.wiki, headers=headers)
    log.info(f"ID of wiki {config.wiki} = {wiki_id}")

    stats = get_all_page_stats(config.url, config.wiki, headers)
    log.info(f"Page view stats: {stats}")


if __name__ == "__main__":
    main()
