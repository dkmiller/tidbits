# Reddit bot via Docker

This code loosely follows
[Build a Reddit Bot](http://pythonforengineers.com/build-a-reddit-bot-part-1/)
in building a Reddit bot which runs in a Docker container. Navigate to
https://www.reddit.com/prefs/apps/ to get the client ID and secret.

## Build and run

Follow:

```powershell
# This is needed to copy any local files.
docker build --tag baby .
docker run baby python baby.py --id <client id> --secret <client secret>
```
