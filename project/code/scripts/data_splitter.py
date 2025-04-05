from abc import ABC, abstractmethod


class DataSplitter(ABC):
    '''
    Abstract class to split dataset into training, test, validation sets
    '''
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass