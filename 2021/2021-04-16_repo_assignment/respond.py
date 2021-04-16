from typing import Tuple
import hydra
import logging
from lxml.html.soupparser import fromstring
from markdown2 import markdown
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


def get_sender(content) -> Tuple[str, str]:
    """
    Return (name, email).
    """
    html = fromstring(content)
    links = html.xpath("//a")
    for link in links:
        from_link = str(link.get("href"))
        from_text = link.text_content()
        if from_link.startswith("mailto:"):
            return (from_text, from_link)

    raise ValueError("Could not find email address.")


def get_folder_id(endpoint: str, filter: str, headers: dict) -> str:
    url = f"{endpoint}me/mailFolders?$filter={filter}"
    log.info(f"GET {url}")
    r = requests.get(url, headers=headers)
    json = r.json()
    rv = json["value"][0]["id"]
    return rv


def respond_to_email(endpoint, email, template: str, headers: dict) -> None:
    email_content = email["body"]["content"]
    (friendly_name, email_address) = get_sender(email_content)
    log.info(f"Sending response to '{friendly_name}' at {email_address}")
    response_md = template.replace("${name}", friendly_name)

    response_html = markdown(response_md)

    message_id = email["id"]
    body = {
        "comment": response_html,
        "toRecipients": [
            {"emailAddress": {"address": "danmill@microsoft.com", "name": "Dan Miller"}}
        ],
    }

    url = f"{endpoint}me/messages/{message_id}/forward"
    log.info(f"POST {url} with body:\n{body}")

    r = requests.post(url, json=body, headers=headers)
    log.info(f"Response: {r} {r.json()}")


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

    for email in emails:
        respond_to_email(config.endpoint, email, config.response_template, headers)


if __name__ == "__main__":
    main()
