import random
from collections import defaultdict
import numpy as np

from core.types import TrainingEntry

def bucket_metric(m: float) -> str:
    if m < 0.33: return 'low'
    if m < 0.66: return 'medium'
    return 'high'


def stratified_sampling_with_features(
        dataset: list[TrainingEntry],
        eval_ratio: float = 0.2,
        seed: int = 42
):
    """
    Stratified sampling ensuring eval set represents all feature combinations.

    Strategy:
    1. Primary stratification by cluster_id (ensures diverse semantic groups)
    2. Within each cluster, balance by explicitness/atomicity/contextuality
    3. Ensure rare combinations appear in both train and eval
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group by cluster first (most important for semantic diversity)
    clusters = defaultdict(list)
    for example in dataset:
        cluster_id = example.metrics.cluster_id
        clusters[cluster_id].append(example)

    train_data = []
    eval_data = []

    print(f"📊 Stratified sampling by features:")
    print(f"   Total clusters: {len(clusters)}")

    for cluster_id, cluster_examples in clusters.items():
        # Within each cluster, further stratify by feature combinations
        feature_groups = defaultdict(list)

        for example in cluster_examples:
            # Create feature signature (discretize continuous values)
            explicitness_bin = bucket_metric(example.metrics.explicitness)
            atomicity_bin = bucket_metric(example.metrics.atomicity)
            contextuality_bin = bucket_metric(example.metrics.contextuality)

            feature_key = f"{explicitness_bin}_{atomicity_bin}_{contextuality_bin}"
            feature_groups[feature_key].append(example)

        # Sample from each feature group within cluster
        cluster_train = []
        cluster_eval = []

        for feature_key, examples in feature_groups.items():
            random.shuffle(examples)

            # Ensure at least 1 example in eval if group has 2+ examples
            if len(examples) == 1:
                cluster_train.append(examples[0])
            else:
                split_idx = max(1, int(len(examples) * (1 - eval_ratio)))
                cluster_train.extend(examples[:split_idx])
                cluster_eval.extend(examples[split_idx:])

        train_data.extend(cluster_train)
        eval_data.extend(cluster_eval)

        print(f"   Cluster {cluster_id}: {len(cluster_train)} train, {len(cluster_eval)} eval")

    # Final shuffle
    random.shuffle(train_data)
    random.shuffle(eval_data)

    return train_data, eval_data
