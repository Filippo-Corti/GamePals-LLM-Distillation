"""
Intent Clustering Library

Clusters LLM commanding inputs by intent and creates evaluation datasets.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

from core.utils.types import LLMCommandingInput, LLMCommandingOutput


def load_jsonl(path: Union[str, Path]) -> list[dict]:
    """Load JSONL file into list of dicts."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: list[dict], path: Union[str, Path]) -> None:
    """Save list of dicts to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def embed_intents(
        intents: list[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
) -> np.ndarray:
    """
    Embed intent strings using sentence transformers.

    Args:
        intents: list of intent strings
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding

    Returns:
        Array of embeddings (n_samples, embedding_dim)
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        intents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings


def cluster_intents(
        embeddings: np.ndarray,
        n_clusters: int,
        seed: int = 42
) -> np.ndarray:
    """
    Cluster embeddings using K-Means.

    Args:
        embeddings: Array of embeddings
        n_clusters: Number of clusters
        seed: Random seed

    Returns:
        Array of cluster IDs
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init="auto"
    )
    return km.fit_predict(embeddings)


def cluster_and_build_dataframe(
        inputs: list[LLMCommandingInput],
        outputs: list[LLMCommandingOutput],
        n_clusters: int = 10,
        per_cluster: Optional[int] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        seed: int = 42,
) -> pd.DataFrame:
    """
    Cluster inputs by intent and build evaluation dataframe.

    Args:
        inputs: list of LLMCommandingInput dataclass objects
        outputs: list of LLMCommandingOutput dataclass objects
        n_clusters: Number of clusters to create
        per_cluster: If provided, sample this many examples per cluster for labeling
        model_name: Sentence transformer model
        seed: Random seed

    Returns:
        DataFrame with columns: input_id, game_state, user_command, chosen_actions,
        latency, intent, reason, cluster_id, selected_for_labeling, [action_types], label
    """
    # Convert dataclasses to dicts

    # Build lookup for inputs by id
    input_lookup = {inp.id: inp for inp in inputs}

    # Extract intents from inputs for clustering
    intent_list = [inp.user_command.command.intent for inp in inputs]

    # Cluster
    print("🔢 Embedding intents...")
    embeddings = embed_intents(
        intent_list,
        model_name=model_name,
    )

    print("🧩 Clustering...")
    cluster_ids = cluster_intents(embeddings, n_clusters=n_clusters, seed=seed)

    # Build dataframe rows
    rows = []
    for output, cluster_id in zip(outputs, cluster_ids):
        input_id = output.input_id
        inp = input_lookup.get(input_id, {})

        row = {
            'input_id': input_id,
            'game_state': inp.game_state.state,
            'user_command': inp.user_command.command,
            'chosen_actions': output.actions,
            'latency': output.latency,
            'intent': inp.user_command.command.intent,
            'reason': output.reason,
            'cluster_id': int(cluster_id),
            'selected_for_labeling': False,
            'action_fully_correct': '',
            'action_unnecessary': '',
            'action_imprecise_sequentiality': '',
            'action_imprecise_parameters': '',
            'action_harming_sequentiality': '',
            'action_harming_parameters': '',
            'action_missing': '',
            'action_harming': '',
            'action_wrong_syntax': '',
            'label': ''
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Mark selected for labeling if per_cluster specified
    if per_cluster is not None:
        print(f"🧪 Selecting {per_cluster} examples per cluster for labeling...")
        selected_indices = []

        for cluster_id in df['cluster_id'].unique():
            cluster_mask = df['cluster_id'] == cluster_id
            cluster_indices = df[cluster_mask].index.tolist()

            if len(cluster_indices) <= per_cluster:
                selected_indices.extend(cluster_indices)
            else:
                np.random.seed(seed)
                sampled = np.random.choice(
                    cluster_indices,
                    size=per_cluster,
                    replace=False
                )
                selected_indices.extend(sampled)

        df.loc[selected_indices, 'selected_for_labeling'] = True

    return df


def save_evaluation_csv(
        df: pd.DataFrame,
        output_path: Union[str, Path],
) -> Path:
    """
    Save evaluation dataframe to CSV.

    Args:
        df: Evaluation dataframe
        output_path: Path to save CSV

    Returns:
        Path where CSV was saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    return output_path


# Convenience function
def run_clustering_pipeline(
        inputs: list[Any],
        outputs: list[Any],
        out_dir: Path,
        n_clusters: int = 10,
        per_cluster: int = 5,
        seed: int = 42
) -> tuple[pd.DataFrame, Path]:
    """
    Complete pipeline: cluster, build dataframe, and save CSVs.

    Args:
        inputs: LLMCommandingInput dataclass objects
        outputs: LLMCommandingOutput dataclass objects
        out_dir: Directory for output files
        n_clusters: Number of clusters
        per_cluster: Examples per cluster for labeling
        seed: Random seed

    Returns:
        Tuple of (dataframe, full_csv_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run clustering
    df = cluster_and_build_dataframe(
        inputs=inputs,
        outputs=outputs,
        n_clusters=n_clusters,
        per_cluster=per_cluster,
        seed=seed,
    )

    # Save full dataset
    full_path = out_dir / "full_dataset_with_clusters.csv"
    save_evaluation_csv(df, full_path)
    print(f"💾 Saved full dataset to {full_path}")

    return df, full_path