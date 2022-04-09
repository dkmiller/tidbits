# Reddit + :hugs:

Fine-tune a language model against top posts in the specified Subreddit.

## Experiments

```
python pipelines/huggingface_reddit.py --config-dir conf --config-name experiments/huggingface_reddit
```

## Links

- [2020-12-23_huggingface](../../2020/2020-12-23_huggingface/generate-text.py)
- [run_clm.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py)
- [Natural Language Generation Part 2: GPT2 and Huggingface](https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a)

## Sample code

Inject Reddit secret into workspace key vault.

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