import uuid
from pathlib import Path
from subprocess import check_output

from cryptography.hazmat.backends import default_backend as crypto_default_backend
from cryptography.hazmat.primitives import serialization as crypto_serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# https://unix.stackexchange.com/a/257648
SSH_CHMOD = 0o600


def write_key(path: Path, content: bytes):
    # https://stackoverflow.com/a/22449476/
    path.write_bytes(content)
    # https://stackoverflow.com/q/57264050/
    # https://stackoverflow.com/a/17776766/
    path.chmod(SSH_CHMOD)


def private_public_key_pair() -> tuple[Path, Path]:
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

    suffix = f".ssh/id_rsa_{uuid.uuid4().hex}"

    private_key_file = Path.home() / suffix
    public_key_file = Path.home() / f"{suffix}.pub"

    write_key(private_key_file, private_key)
    write_key(public_key_file, public_key)

    # https://stackoverflow.com/a/56592179/
    # It's not enough to naively replace "PRIVATE KEY" with "RSA PRIVATE KEY":
    # https://www.diffchecker.com/g7UNtu8E/
    check_output(["ssh-keygen", "-p", "-m", "PEM", "-f", private_key_file, "-N", ""])

    return (private_key_file, public_key_file)
