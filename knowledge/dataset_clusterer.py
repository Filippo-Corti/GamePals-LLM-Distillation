from typing import Callable, Any, Iterable
from sklearn.cluster import AgglomerativeClustering
import numpy as np

from datasets import GamePalsDatasetTransformer, GamePalsDataset


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
        features = np.array([self.to_features(item) for item in x])

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1e-2,
            linkage='average'
        )
        labels = clustering.fit_predict(features)

        new_x = list()
        for cluster_id in set(labels):
            if cluster_id == -1: continue

            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_features = features[cluster_indices]

            # Compute centroid
            centroid = cluster_features.mean(axis=0)

            # Find closest item to centroid
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            center_idx = cluster_indices[np.argmin(distances)]

            new_x.append(x[center_idx])

        return GamePalsDataset(new_x)

