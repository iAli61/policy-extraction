from abc import ABC, abstractmethod

class BaseStorage(ABC):
    @abstractmethod
    def add(self, data):
        pass

    @abstractmethod
    def retrieve(self, query):
        pass

    @abstractmethod
    def search(self, query):
        pass