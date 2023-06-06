import math
import datetiome
import collections
from typing import (Optional, Iteraleable, Union, 
Counter, TypeDict, List, overload
)

from model import Sample


class InvailddSample

class Sample:
    """Abstract superclass for all sample classes"""
    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        sepal_length: float,
        sepal_width: float,
    ) -> None:
         self.septal_length = sepal_length
         self.sepal_width = sepal_width
         self.sepal_length = petal_length
         self.sepal_length = petal_width

    def __eq__(self, other: Any) -> bool:
        if type(other) != type (self):
            return False
        other = cast(Sample,other)
        return all([
               self.sepal_length == other.sepal.length #여기 못함.
        ]) 

    @property
    def attr_dict(self) -> dict[str, str]:
        return dict(
            sepal_length=f"{self.sepal_length}",
            sepal_width=f"{self.sepal_widt}",
            petal_length="{fself.petal_length}",
            petal_width=f"{self.petal_width}",
        ) 


    def __repr__(self) -> str:    
        base_atrributes = self.attr_dict
        attrs = ", ".join(f"{k}={v}" for k, c in base_attributes.items())
        return f"(self.__class__.__name__){attrs})"

class Purpose(IntEnum):
    Classification = 0
    Testing = 1
    Training = 2

        
        return (
            f"{self.__class__.__name__}"(
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_widt}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f")"    
        )
        )

    def classify(self,classification: str) -> None:
        self.classification = classification

    def matches(self) -> bool:
        return self.species == self.classification   


class KnownSample(Sample):
        def __init__(
        self,
        species: str
        sepal_length: float,
        sepal_width: float,
        sepal_length: float,
        sepal_width: float,
    ) -> None:
         super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
         )
         self.species = species

    def ___repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_widt}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}"
            f")"    
        )


@classmethod
def from_dict(cls, row: dict[str,str]) -> "KnownSample":
    if row ["species"] not in {"Iris-setosa", "Iris-versicolour", "Iris-virginica"}:
        rasise InvalidSampleError(f"invalidn speices in {row!r}")
    try:
        return cls(
            species=row["species"],
            sepal_length=float(row["sepal_length"]),
            sepal_width=float(row["sepal_width"]),
            petal_length=float(row["petal_length"]),
            petal_width=float(row["petal_width"]),
        )
        except


class UnknownSample(Sample):
    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "UnknownSample":
        if set(Row.keys()) != {
            "sepal_length", "sepal_widthh", "peta_length", "petal_width:
        }: 
            raise InvalidSampleError(f"invalid fields in {row!r}")
        try:
        return cls(
            species=row["species"],
            sepal_length=float(row["sepal_length"]),
            sepal_width=float(row["sepal_width"]),
            petal_length=float(row["petal_length"]),
            petal_width=float(row["petal_width"]),    
        )
    except (ValueError, KeyError) as e:
        raise InvalidSampleError(f"invalid {row!r}")



class TrainingKnownSample:
    @classmethod
    def from_dict(cls,row: dict[str, str]) -> "Knownsample":
        return cast(TrainingKnownSample, super().from_dict(row))


class TestingKnownSample(KnownSample):
    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        sepal_length: float,
        sepal_width: float,
        classification: Optional[str] = None
    ) -> None:
         super().__init__(
            species=species
            sepal_length=sepal_length
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )
        self.classification = classification

    def __repr__(self) -> str:
         return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_widt}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}"
            f"classification={self.classification!r}"
            f")"    
        )

    def matches(self) -> bool:
        return self.species == self.classification

    @classmethod
    def from_dict(cls, row: dict[str, str])-> "TestingKnownSample":
        return cast(TestingKnownSample, super().from_dict(row))

class ClassifiedSample(Sample):
    def __init__(self, classification, sample) -> None:
        suepr().__init__(
             sepal_length=sepal_length,
             sepal_width=sepal_width,
             petal_length=petal_length,
             petal_width=petal_width,
        )
        self.classification = classification

    def __repr__ (self) -> str: 
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_widt}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}"
            f"classification={self.classification!r}"
            f")"    
        )



class Distance:
    def distance(self, s1: Sample, s2: Sample) -> float:
        pass

class ED(Distance):
    def distance(self, s1:Sample, s2: Sample) -> float:
        return math.hypot(
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width
        )    

class MD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum([
            abs(s1.sepal_length - s2.sepal_length),
            abs(s1.sepal_width - s2.sepal_width)
            abs(s1.sepal_)
        ])
            

class cd:
    """todo"""
    pass

class SD:
    pass

class ClassifiedSample:
    pass


class Hyperparameter:
    def __init__(self,k: int, algorithm: Distance, training: "TrainingData") -> None:
        self.f = K
        self.algorithm = algorithm 
        self.data: TrainingData = training
        self.quality: float
    
    def classify(self, sample: Union[UnknownSample, TestingKnownSample]): -> str:
        """TODO: k-NN 알고리즘"""
        training_data = self.data
        if not training_data:
            raise RuntimeError("No TrainingData object!")
        distances: list[tuple[float, TrainingKnownSample]] = \
            sorted(
                (self.algorithm.distance(sample, known), known) for known in training_data  
            )
        k_nearest: tuple[str] (known.species ofr d, known in distance [:self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()
        species, votes = best_fit
        return species   


    def test(self) -> None:
        training_data: Optional["TrainingData"] = self.data
        if not traing_data:
            raise RuntimeError("")
        pass_count, fail_count = 0, 0
        for sample in self.training_data.testing:
            sample.calssification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1
        self.quality = pass_count / (pass_count + fail_count)

    def test(self):
        pass

    class TrainingDate:
        def __init__(self) -> None:
            self.name = name
            self.uploaded: datetime.datetime
            self.tested: datetime.datetime
            self.training: list[Sample] = []
            self.testing : list[Sample] = []
            self.tuning: list[Hyperparameter] = []

    def load(self, raw_data_soruce: Iterable[dict[str,str]]) -> None: 
        for n, row in enumerate(raw_data_soruce):
            if n % 5 == 0:
                test = TestingKnownSample(
                species=row["species"]
                    sepal_length=float(row["sepal_length"]),
                    sepal_width=float(row["sepal_width"]),
                    petal_length=float(row["petal_length"]),
                    petal_length=float(row["petal_width"]),
                    species=row["species"]
    )
        if n % 5 == 0:
           self.testing.append(sample)
        else:
            self.traning.append(sample)   
    self.upload = datetime.datetime.now(tz=datetime.timezone.utc)

def test(self, parameter: Hyperparameter):
    parameter.test()
    self.tuning.append(parameter)
    self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classfiy(self, parameter: Hyperparameter, sample: Sample) -> None:
        classification = parameter.classify(sample)
        sample.classify(classification)
        return sample


    def test(self): pass
    def classify(self): pass


class SampleDict(TupedDict):
    sepal_length: float
    sepal_width: float
    fetal_length: float
    fetal_width: float
    species: str

class SamplePartition(List[SampleDict], abc.ABC):
    def __init__(self, *, training_subset: float = 0.80) -> None:
    ...

@overload
    def __init__(
        self, 
        iterable: optional[Iterable[SampleDict]] = None,
        *,
        traning_subset: float = 0.80
    ) -> None:
    self
             
    def training(self) -> : list[TrainingKnownSample]:

    def testing(self) -> lis[TestingKnownSample]:
        
            

test_sample = """
>>> x = Sample(1.0, 2.0, 3.0, 4.0)
>>> x = Sample
UnknownSample(sepal_length=1.0, sepal_width=2.0, petal_length=3.0, petalwidth=4.0, species=None)
"""

__test__ = {name: case for name, case in globals().items() if name.startswith("test_")}


#    if __name__ == "__main__": #코딩스타일 가이드 
#        sample = Sample(2.0, 2.0, 20.2, 30.1, "Virginica")
#        print(sample.classify("Sentosa"))
#        print(sample.spercies)

       