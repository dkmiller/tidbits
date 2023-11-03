from dataclasses import dataclass


@dataclass
class SshHost:
    host: str
    port: int
    user: str
