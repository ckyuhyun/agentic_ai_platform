
from enum import Enum

class WeaviateStatusCode(str, Enum):
    NoCollection = "NoCollection" # if no collection is pointed out
    NoExistCollection = "NoCollectionExisted" # if collection is not existed
