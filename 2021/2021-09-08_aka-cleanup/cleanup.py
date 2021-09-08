from argparse import ArgumentParser
import pandas as pd
import requests


def main(args):
    link_info = pd.read_csv(args.links)
    vanity_urls = list(link_info["VanityUrl"])

    headers = {"Authorization": f"Bearer {args.token}"}

    for url in vanity_urls:
        print(f"Deleting: {url}")
        # https://microsoft.sharepoint.com/teams/RedirectionServiceInfo/SitePages/Redirection%20Service%20API%20Guide.aspx
        api_url = (
            f"https://redirectionapi.trafficmanager.net/api/aka/1/ust/harddelete/{url}"
        )
        r = requests.delete(api_url, headers=headers)
        print(f"\t{r}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--links", required=True)
    args = parser.parse_args()

    main(args)
