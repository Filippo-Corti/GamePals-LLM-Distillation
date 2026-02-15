from abc import ABC, abstractmethod

from src.core.datasets import GamePalsDataset


class GamePalsDatasetTransformer(ABC):

    @abstractmethod
    def transform(self, x: GamePalsDataset) -> GamePalsDataset:
        pass

