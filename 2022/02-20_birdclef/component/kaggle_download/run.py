from argparse_dataclass import dataclass
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import logging
import os
from pathlib import Path
import zipfile


# You can't even import `KaggleApi` without these defined :/
# https://stackoverflow.com/a/4028943
if not (Path.home() / ".kaggle/kaggle.json").exists():
    os.environ["KAGGLE_USERNAME"] = ""
    os.environ["KAGGLE_KEY"] = ""

from kaggle.api import KaggleApi


log = logging.getLogger(__name__)


@dataclass
class Config:
    output: str
    competition: str = "birdclef-2022"
    user: str = "antifragilista"
    vault_url: str = "https://danmill-kv.vault.azure.net/"
    secret_name: str = "kaggle-api-key"


def get_kaggle_api_from_azure(vault_url: str, secret_name: str, user: str) -> KaggleApi:
    # WTF... why is this necessary.
    cred = DefaultAzureCredential(
        managed_identity_client_id="fc650f41-009d-4934-9662-699f2bc6d9b0"
    )
    vault_client = SecretClient(vault_url, cred)
    api_key = vault_client.get_secret(secret_name).value

    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"] = api_key

    from kaggle.api import KaggleApi

    rv = KaggleApi()
    rv.authenticate()
    return rv


def main(config: Config):
    kaggle_client = get_kaggle_api_from_azure(
        config.vault_url, config.secret_name, config.user
    )
    kaggle_client.competition_download_files(
        config.competition, config.output, quiet=False
    )

    zip_path = next(Path(config.output).glob("*.zip"))
    log.info(f"Unzipping {zip_path} to {config.output}")
    # https://stackoverflow.com/a/3451150
    with zipfile.ZipFile(zip_path, "r") as zip:
        zip.extractall(config.output)

    log.info(f"Done unzipping, deleting {zip_path}")
    zip_path.unlink()


if __name__ == "__main__":
    config = Config.parse_args()  #  type: ignore
    logging.basicConfig(level="INFO")
    main(config)
