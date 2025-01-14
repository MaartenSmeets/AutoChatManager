import shelve
import os
import hashlib

class CacheManager:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path

    def _hash_key(self, prompt: str, model_name: str) -> str:
        key = f"{model_name}:{prompt}"
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    def get_cached_response(self, prompt: str, model_name: str):
        key = self._hash_key(prompt, model_name)
        with shelve.open(self.cache_path) as db:
            if key in db:
                return db[key]
        return None

    def store_response(self, prompt: str, model_name: str, response: str):
        key = self._hash_key(prompt, model_name)
        with shelve.open(self.cache_path) as db:
            db[key] = response
