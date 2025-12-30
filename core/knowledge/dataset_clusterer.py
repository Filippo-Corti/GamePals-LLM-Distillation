from typing import Callable, Any, Iterable
from sklearn.cluster import DBSCAN
import numpy as np

from core.datasets import GamePalsDatasetTransformer, GamePalsDataset


class DatasetClusterer(GamePalsDatasetTransformer):
    """
    A GamePalsDatasetTransformers specialized to cluster items in the dataset
    and only keep the cluster centers
    """

    def __init__(
            self,
            to_features: Callable[[Any], Iterable],
    ):
        """
        Creates a DatasetClusterer

        :param to_features: the function that transforms each item in the dataset into its features vector
        """
        self.to_features = to_features

    def transform(self, x: GamePalsDataset) -> GamePalsDataset:
        """
        Reduces the dataset to only its cluster centers

        :param x: a gamepals dataset
        :return: the new gamepals dataset
        """
        features = np.array([self.to_features(item) for item in x], dtype=np.float32)

        clustering = DBSCAN(
            eps=1e-2,
            min_samples=1,
            metric="euclidean",
        )
        labels = clustering.fit_predict(features)

        new_x = GamePalsDataset()
        for cluster_id in set(labels):
            if cluster_id == -1: continue

            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_features = features[cluster_indices]

            # Compute centroid
            centroid = cluster_features.mean(axis=0)

            # Find the closest item to centroid
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            center_idx = cluster_indices[np.argmin(distances)]

            new_x.append(x[center_idx])

        return new_x
