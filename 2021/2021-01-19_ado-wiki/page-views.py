from typing import Dict, List, Union
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


@hydra.main(config_name="config")
def main(config):
    log.info(f"Configuration: {config}")

    headers = get_auth_headers(*config.scopes)

    r = requests.get(f"{config.url}/_apis/wiki/wikis", headers=headers)
    wikis = r.json()
    wiki_id = jmespath.search(f"value[?name == '{config.wiki}'].id", wikis)[0]

    r = requests.get(f"{config.url}/_apis/wiki/wikis/{wiki_id}/pages?path=/Home&api-version=6.0", headers=headers)
    page_id = r.json()["id"]

    # https://dev.azure.com/msdata/Vienna/_apis/wiki/wikis/0003dacc-7379-491b-901d-416322f17283/pages?recursionLevel=full&api-version=6.0

    # With wiki name vs. ID:
    # https://dev.azure.com/msdata/Vienna/_apis/wiki/wikis/aml-1p-onboarding/pages?recursionLevel=full&api-version=6.0

    # page view stats...
    # https://dev.azure.com/msdata/Vienna/_apis/wiki/wikis/0003dacc-7379-491b-901d-416322f17283/pages/11188/stats?pageViewsForDays=30


if __name__ == "__main__":
    main()
