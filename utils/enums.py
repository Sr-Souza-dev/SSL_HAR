from enum import Enum

class Datas(Enum):
    PAMAP = ("pamap", "dat", " ", "https://dl.dropboxusercontent.com/scl/fi/8bjtqgoqrfuzn2c9s1vu0/pamap.zip?rlkey=5el9xga9hpvrsh2ysebwibgmm&st=rr9f9lge&dl=0")
    MOBIT = ("mobit", "dat", " ", "https://dl.dropboxusercontent.com/scl/fi/jgfknfd9wxh0ryy6beb7b/mobit.zip?rlkey=9qr5lczjze6dhhx7vpuq2z9uz&st=0edjkwvk&dl=0")
    MOTION = ("motion", "dat", " ", "https://dl.dropboxusercontent.com/scl/fi/fcucqw5u3zlqy6c22aglb/motion.zip?rlkey=0xzawzga0rhzk2095fgrywqrx&st=t507y88c&dl=0")
    HAR = ("har", "csv", ",", "https://www.ic.unicamp.br/~edson/disciplinas/mo436/2024-1s/data/har.zip")

    def __init__(self, value, type, sep, url):
        self._value_ = value
        self._type = type
        self._sep = sep
        self._url = url
    @property
    def url(self):
        return self._url
    
    @property
    def type(self):
        return self._type
    
    @property
    def sep(self):
        return self._sep

class Sets(Enum):
    TEST = "test"
    TRAIN = "train"
    VALIDATION = "validation"
    REAL = "real"
    PREDICTION = "prediction"

class ModelTypes(Enum):
    PRETEXT = "pretext"
    DOWNSTREAM = "downstream"

main_data = Datas.MOBIT
teste_size = 0
