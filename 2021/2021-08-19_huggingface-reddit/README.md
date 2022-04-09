# Reddit + :hugs:

Fine-tune a language model against top posts in the specified Subreddit.

## Experiments

```
python pipelines/huggingface_reddit.py --config-dir conf --config-name experiments/huggingface_reddit
```

## Components

```
python components/download_reddit_data/run.py  --output-directory ~/tmp --client-id XnkMHEYUujv1wA7EkmToWg --client-secret $client_secret --post-limit 100 --subreddits news,funny --top-mode all
```

```
python components/prepare_json_data/run.py  --input-directory ~/tmp/ --source-jsonpaths $.title $.selftext --source-key title_and_text --target-key n_gilds --output-directory ~/tmp2/ --output-file-name data.json --target-jsonpath '$.gilded'
```

## Links

- [2020-12-23_huggingface](../../2020/2020-12-23_huggingface/generate-text.py)
- [run_clm.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py)
- [Natural Language Generation Part 2: GPT2 and Huggingface](https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a)
- https://github.com/huggingface/transformers/pull/8062

## Sample code

Inject Reddit secret into workspace key vault. TODO: move that to pipeline
code or a "misc" script.

```python
from azure.identity import AzureCliCredential
import requests

cred = AzureCliCredential()

token = cred.get_token("https://vault.azure.net/.default")
headers = {"Authorization": f"Bearer {token.token}"}

keyvault = "amlkeyvault6upghgpmdpxoq"

client_secret = "<your client secret>"

r = requests.put(
    f"https://{keyvault}.vault.azure.net/secrets/reddit-client-secret?api-version=7.2",
    json={"value": client_secret},
    headers=headers,
)
print(r.content)
```