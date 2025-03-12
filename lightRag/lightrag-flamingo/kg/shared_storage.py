# filepath: /home/azureuser/policy-extraction/lightRag/lightrag/kg/shared_storage.py
class SharedStorage:
    def __init__(self):
        self.storage = {}

    def insert(self, key, value):
        self.storage[key] = value

    def retrieve(self, key):
        return self.storage.get(key, None)

    def delete(self, key):
        if key in self.storage:
            del self.storage[key]

    def clear(self):
        self.storage.clear()

    def get_all(self):
        return self.storage.items()