# Interact with Outlook Calendar using the Microsoft Graph API

Title says it all. Here's a starting point (sadly: size above 25 is not
respected).

> POST https://graph.microsoft.com/v1.0/search/query

```json
{
    "requests": [
        {
            "entityTypes": [
                "event"
            ],
            "query": {
                "queryString": "\"AML DS - feedback & support DRI\""
            }
        }
    ]
}
```

Next:

> GET https://graph.microsoft.com/v1.0/me/events/AAMkADE1YWU3Y2JiLWViMWYtNGEwYi1hYTc0LWY3OGYzZTczMmI3NwBGAAAAAACM8gF7+BvjQ6KB8i6LxXPmBwBO+KBBn5icQ6fCqpIQYqzlAAAAAAENAABO+KBBn5icQ6fCqpIQYqzlAANCLYsaAAA=?$select=subject,body,bodyPreview,organizer,attendees,start,end,location

## Links

- [Use the Microsoft Search API to search calendar events](https://docs.microsoft.com/en-us/graph/search-concept-events)
- [Use the Microsoft Graph SDKs to batch requests](https://docs.microsoft.com/en-us/graph/sdks/batch-requests?tabs=csharp)
- [Microsoft Graph permissions reference](https://docs.microsoft.com/en-us/graph/permissions-reference)
