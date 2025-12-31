from abc import ABC, abstractmethod

from core.datasets import GamePalsDataset


class GamePalsDatasetTransformer(ABC):

    @abstractmethod
    def transform(self, x: GamePalsDataset) -> GamePalsDataset:
        pass

