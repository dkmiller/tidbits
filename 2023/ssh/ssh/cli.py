from ssh.rsa import private_public_key_pair


def gen_rsa():
    private, _ = private_public_key_pair()
    hex = private.stem.split("_")[-1]
    print(hex)
