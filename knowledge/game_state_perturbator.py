from typing import Callable

from datasets import GamePalsDatasetTransformer


class GameStatePerturbator(GamePalsDatasetTransformer):
    """
    TODO
    """

    def __init__(
            self,
            generate_perturbations: Callable[[dict], dict],
    ):
        pass

    def transform(self, x: "GamePalsDataset") -> "GamePalsDataset":
        pass
