import argparse
from typing import Tuple
from azure.identity import DefaultAzureCredential
import requests


def parse_event_request(r) -> Tuple[list, str]:
    json = r.json()
    next = json.get("@odata.nextLink")
    events = json.get("value")
    return (events, next)


def get_event_id(headers):
    r = requests.get("https://graph.microsoft.com/v1.0/me/calendar/events", headers=headers)
    events, next = parse_event_request(r)

    while next:
        for event in events:
            if "AML DS - feedback" in str(event):
                return event["id"]
        print(f"Calling {next}")
        r = requests.get(next, headers=headers)
        events, next = parse_event_request(r)

    raise Exception(f"Unable to find reference event.")
        

    # for event in events:
    #     if "AML DS - feedback" in str(event):
    #         print(event)
    #         next = None

    # while next:
    #     print(f"Calling {next}")
    #     r = requests.get(next, headers=headers)
    #     json = r.json()
    #     next = json.get("@odata.nextLink")
    #     print(json.keys())
    #     if "error" in json:
    #         print(json["error"])
    #     new_events = json["value"]
    #     for event in new_events:
    #         if "AML DS - feedback" in str(event):
    #             print(event)
    #             next = None

    #     events.extend(new_events)



def main(args):
    if not args.token:
        token = DefaultAzureCredential().get_token("https://graph.microsoft.com/").token
    else:
        token = args.token
    headers = {"Authorization": f"Bearer {token}"}


    if not args.event_id:
        event_id = get_event_id(headers)
    else:
        event_id = args.event_id

    print(event_id)

    # r = requests.get("https://graph.microsoft.com/v1.0/me/calendar/events?$top=100&$skip=10", headers=headers)
    # json = r.json()
    # next = json.get("@odata.nextLink")

    # events = r.json()["value"]

    # for event in events:
    #     if "AML DS - feedback" in str(event):
    #         print(event)
    #         next = None

    # while next:
    #     print(f"Calling {next}")
    #     r = requests.get(next, headers=headers)
    #     json = r.json()
    #     next = json.get("@odata.nextLink")
    #     print(json.keys())
    #     if "error" in json:
    #         print(json["error"])
    #     new_events = json["value"]
    #     for event in new_events:
    #         if "AML DS - feedback" in str(event):
    #             print(event)
    #             next = None

    #     events.extend(new_events)

    # print(events.json().keys())

    # j = events.json()["value"]
    # for event in j:
    #     if "AML DS - feedback" in str(event):
    #         print(event)
    #         print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_id", type=str)
    parser.add_argument("--token", type=str)
    args = parser.parse_args()
    main(args)
