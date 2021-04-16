import hydra
import logging
import requests


log = logging.getLogger(__name__)


def get_emails(endpoint: str, filter: str, folder: str, headers: dict) -> list:
    url = f"{endpoint}me/messages?$filter={filter}"
    log.info(f"GET {url}")
    r = requests.get(url, headers=headers)
    json = r.json()

    rv = []
    for email in json["value"]:
        if email["parentFolderId"] == folder:
            rv.append(email)

    return rv


def get_folder_id(endpoint: str, filter: str, headers: dict) -> str:
    url = f"{endpoint}me/mailFolders?$filter={filter}"
    log.info(f"GET {url}")
    r = requests.get(url, headers=headers)
    json = r.json()
    rv = json["value"][0]["id"]
    return rv


@hydra.main(config_name="config")
def main(config):
    log.info(f"Configuration: {config}")

    access_token = config.access_token
    headers = {"Authorization": f"Bearer {access_token}"}
    log.info(f"Headers: {headers}")

    inbox_id = get_folder_id(config.endpoint, config.folders_filter, headers)
    log.info(f"Got inbox: {inbox_id}")

    emails = get_emails(config.endpoint, config.messages_filter, inbox_id, headers)
    log.info(f"Got {len(emails)} email(s)")


if __name__ == "__main__":
    main()
