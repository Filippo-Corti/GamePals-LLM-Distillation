from abc import ABC, abstractmethod


class GamePalsDatasetTransformer(ABC):

    @abstractmethod
    def transform(self, x: "GamePalsDataset") -> "GamePalsDataset":
        pass

