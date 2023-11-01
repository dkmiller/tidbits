from pytest import fixture
from cryptography.hazmat.primitives import serialization as crypto_serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend as crypto_default_backend
import tempfile
import uuid
from pathlib import Path


@fixture
def key_pair():
    # https://stackoverflow.com/a/39126754/

    key = rsa.generate_private_key(
        backend=crypto_default_backend(), public_exponent=65537, key_size=2048
    )

    private_key = key.private_bytes(
        crypto_serialization.Encoding.PEM,
        crypto_serialization.PrivateFormat.PKCS8,
        crypto_serialization.NoEncryption(),
    )

    public_key = key.public_key().public_bytes(
        crypto_serialization.Encoding.OpenSSH, crypto_serialization.PublicFormat.OpenSSH
    )

    suffix = f"id_rsa_{uuid.uuid4().hex}"

    private_key_file = Path.home() / suffix
    public_key_file = Path.home() / f"{suffix}.pub"

    private_key_file.write_bytes(private_key)
    public_key_file.write_bytes(public_key)

    try:
        yield {
            "public": public_key_file.absolute(),
            "private": private_key_file.absolute(),
        }
    except:
        private_key_file.unlink()
        public_key_file.unlink()
