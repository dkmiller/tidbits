Imitate

https://github.com/dkmiller/tidbits/blob/master/2022/06-29_azdo-conns/add-to-all-subscriptions.py


https://patch-diff.githubusercontent.com/raw/airbnb/lottie-web/pull/2971.diff

https://api.github.com/repos/airbnb/lottie-web/pulls/2971

https://github.com/orgs/community/discussions/24460

https://api.github.com/repos/airbnb/lottie-web/pulls/2971

https://api.github.com/repos/airbnb/lottie-web/pulls/2971/comments

https://api.github.com/repos/airbnb/lottie-web/pulls/comments

https://docs.github.com/en/rest/pulls/comments?apiVersion=2022-11-28

https://github.com/airbnb/lottie-web/pull/2971

https://github.com/orgs/airbnb/repositories?language=&q=&sort=&type=public

```bash
/usr/bin/python3 run.py --token $(op item get 'GitHub read-only' --field credential)
```

https://docs.google.com/document/d/1WmhNle2aoLqFJhUhkCkuiB3o0czJCPOi81hH1pAOJfM/

## Consumption

```python
import pandas as pd
from unidiff import PatchSet


df = pd.read_json("output.jsonl", lines=True)

# https://stackoverflow.com/a/21295630
# Max 2507, seems fine.
df.completion.str.len().max()



long_prompt = df[df["prompt"].str.len() == 38581248].iloc[0]["prompt"]

patch = PatchSet(long_prompt[126:])
```