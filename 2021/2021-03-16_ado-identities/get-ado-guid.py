import argparse
from azure.identity import DefaultAzureCredential
import logging
import requests


log = logging.getLogger(__name__)


def main(args):
    log.info(f"Arguments: {args}")

    token = DefaultAzureCredential().get_token(
        "499b84ac-1321-427f-aa17-267ca6975798/.default"
    )
    headers = {"Authorization": f"Bearer {token.token}"}
    log.info("Got authentication header")

    # Strangely, this works for some names but not others.
    body = {"query": args.query, "subjectKind": ["User"]}
    r = requests.post(
        f"https://vssps.dev.azure.com/{args.org}/_apis/graph/subjectquery?api-version={args.api_version}",
        json=body,
        headers=headers,
    )
    json = str(r.json())
    log.info(f"Results: {json[:100]}...")

    body = {
        "query": args.query,
        "identityTypes": ["user", "group"],
        "operationScopes": ["ims", "source"],
    }
    r = requests.post(
        f"https://dev.azure.com/{args.org}/_apis/IdentityPicker/Identities?api-version={args.api_version}",
        json=body,
        headers=headers,
    )
    json = r.json()
    log.info(f"Results: {json}")
    log.info(f"Found {len(json['results'])} results")
    for result in json["results"]:
        id = result["identities"][0]["localId"]
        log.info(f"ID: {id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-version", default="6.0-preview")
    parser.add_argument("--org", default="msdata")
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level="INFO")
    main(args)
