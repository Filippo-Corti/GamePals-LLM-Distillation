from typing import Callable, Any, Iterable

from core.datasets import GamePalsDatasetTransformer, GamePalsDataset


class DatasetPerturbator(GamePalsDatasetTransformer):
    """
    A GamePalsDatasetTransformers specialized to apply small
    perturbations to a dataset.
    """

    def __init__(
            self,
            perturbate: Callable[[Any], Iterable],
    ):
        """
        Creates a DatasetPerturbator

        :param perturbate: the function that, given an item, produces its perturbations
        """
        self.perturbate = perturbate

    def transform(self, x: GamePalsDataset) -> GamePalsDataset:
        """
        Enlarges the dataset with the perturbations of its items

        :param x: a gamepals dataset
        :return: the new gamepals dataset
        """
        new_x = GamePalsDataset()
        for item in x:
            perturbations = self.perturbate(item)
            for p in perturbations:
                new_x.append(p)
        return new_x
