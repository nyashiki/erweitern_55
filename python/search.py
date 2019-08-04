from abc import ABCMeta, abstractmethod

class Search(metaclass=ABCMeta):
    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def best_move(self):
        raise NotImplementedError

    @abstractmethod
    def dump(self):
        raise NotImplementedError
