from cachelib import FileSystemCache
from injector import Module, provider, singleton


class CacheClient(Module):
    @singleton
    @provider
    def cache(self) -> FileSystemCache:
        return FileSystemCache(".ptah")
