import os


def docker_images():
    raw = os.environ["WORKSPACE_DOCKER_IMAGES"]
    uris = [uri.strip() for uri in raw.split(" ") if uri]
    return {
        uri.split(":")[0]: uri
        for uri in uris
    }
