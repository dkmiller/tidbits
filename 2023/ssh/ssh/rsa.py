import os
import uuid
from pathlib import Path

import socket
import threading
import paramiko
from cryptography.hazmat.backends import default_backend as crypto_default_backend
from cryptography.hazmat.primitives import serialization as crypto_serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pathlib import Path



# https://unix.stackexchange.com/a/257648
SSH_CHMOD =  0o600


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


    # https://stackoverflow.com/a/22449476/
    private_key_file.write_bytes(private_key)
    # https://stackoverflow.com/q/57264050/
    # https://stackoverflow.com/a/17776766/
    private_key_file.chmod(SSH_CHMOD)
    # os.chmod(private_key_file, SSH_CHMOD)

    # https://stackoverflow.com/a/56592179/
    os.system(f"ssh-keygen -p -m PEM -f {private_key_file} -N ''")

    public_key_file.write_bytes(public_key)
    os.chmod(public_key_file, SSH_CHMOD)

    return (private_key_file, public_key_file)
