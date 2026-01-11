"""
Intent Clustering Library

A module for clustering and analyzing intent data from LLM outputs.
Provides functions for embedding, clustering, and preparing evaluation datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


# -----------------------------
# IO Functions
# -----------------------------

def load_jsonl(path: Union[str, Path]) -> List[Dict]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


# -----------------------------
# Embedding + Clustering
# -----------------------------

def embed_intents(
        examples: List[Dict],
        intent_field: str = "intent",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress: bool = True
) -> np.ndarray:
    """
    Embed the intent field from examples using sentence transformers.

    Args:
        examples: List of dictionaries containing intent data
        intent_field: Name of the field containing intent text
        model_name: Name of the sentence transformer model to use
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar

    Returns:
        NumPy array of embeddings (n_samples, embedding_dim)
    """
    model = SentenceTransformer(model_name)
    intents = [ex[intent_field] for ex in examples]

    embeddings = model.encode(
        intents,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True
    )
    return embeddings


def cluster_intents(
        embeddings: np.ndarray,
        n_clusters: int,
        seed: int = 42,
        **kmeans_kwargs
) -> np.ndarray:
    """
    Cluster embeddings using K-Means.

    Args:
        embeddings: Array of embeddings to cluster
        n_clusters: Number of clusters to create
        seed: Random seed for reproducibility
        **kmeans_kwargs: Additional arguments to pass to KMeans

    Returns:
        Array of cluster IDs (one per example)
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init="auto",
        **kmeans_kwargs
    )
    return km.fit_predict(embeddings)


def add_cluster_labels(
        examples: List[Dict],
        cluster_ids: np.ndarray,
        cluster_field: str = "intent_cluster"
) -> List[Dict]:
    """
    Add cluster IDs to examples in-place.

    Args:
        examples: List of example dictionaries
        cluster_ids: Array of cluster assignments
        cluster_field: Field name for cluster ID

    Returns:
        Updated examples list (modified in-place)
    """
    for ex, cid in zip(examples, cluster_ids):
        ex[cluster_field] = int(cid)
    return examples


# -----------------------------
# Subset Extraction
# -----------------------------

def extract_balanced_subset(
        examples: List[Dict],
        per_cluster: int,
        cluster_field: str = "intent_cluster",
        seed: int = 42
) -> pd.DataFrame:
    """
    Extract a balanced subset of examples from each cluster.

    Args:
        examples: List of example dictionaries with cluster assignments
        per_cluster: Maximum number of examples to sample per cluster
        cluster_field: Name of the cluster ID field
        seed: Random seed for sampling

    Returns:
        DataFrame containing the balanced subset
    """
    df = pd.DataFrame(examples)

    rows = []
    for cluster_id, group in df.groupby(cluster_field):
        if len(group) <= per_cluster:
            rows.append(group)
        else:
            sampled = group.sample(n=per_cluster, random_state=seed).copy()
            rows.append(sampled)

    subset = pd.concat(rows, ignore_index=True)
    return subset


# -----------------------------
# CSV Preparation
# -----------------------------

def prepare_manual_eval_csv(
        df: pd.DataFrame,
        rubric_columns: Optional[Dict[str, str]] = None,
        keep_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare a DataFrame for manual evaluation with rubric columns.

    Args:
        df: Input DataFrame
        rubric_columns: Dict mapping rubric column names to default values
        keep_columns: List of columns to keep in output (None = keep all)

    Returns:
        DataFrame with rubric columns added and filtered
    """
    df = df.copy()

    # Default rubric columns
    if rubric_columns is None:
        rubric_columns = {
            "rubric_correctness": "",
            "rubric_redundancy": "",
            "rubric_order": "",
            "rubric_notes": ""
        }

    # Add rubric columns if they don't exist
    for col, default in rubric_columns.items():
        if col not in df.columns:
            df[col] = default

    # Filter columns if specified
    if keep_columns is not None:
        available_cols = [c for c in keep_columns if c in df.columns]
        df = df[available_cols]

    return df


# -----------------------------
# Pipeline Functions
# -----------------------------

def cluster_and_sample_pipeline(
        examples: List[Dict],
        n_clusters: int = 10,
        per_cluster: int = 5,
        intent_field: str = "intent",
        cluster_field: str = "intent_cluster",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        seed: int = 42,
        show_progress: bool = True
) -> tuple[List[Dict], pd.DataFrame]:
    """
    Complete pipeline: embed, cluster, and extract balanced subset.

    Args:
        examples: List of example dictionaries
        n_clusters: Number of clusters to create
        per_cluster: Examples to sample per cluster
        intent_field: Field containing intent text
        cluster_field: Field name for cluster assignments
        model_name: Sentence transformer model name
        seed: Random seed
        show_progress: Whether to show progress bars

    Returns:
        Tuple of (clustered_examples, subset_dataframe)
    """
    # Embed
    embeddings = embed_intents(
        examples,
        intent_field=intent_field,
        model_name=model_name,
        show_progress=show_progress
    )

    # Cluster
    cluster_ids = cluster_intents(embeddings, n_clusters=n_clusters, seed=seed)

    # Add labels
    clustered_examples = add_cluster_labels(
        examples,
        cluster_ids,
        cluster_field=cluster_field
    )

    # Extract subset
    subset_df = extract_balanced_subset(
        clustered_examples,
        per_cluster=per_cluster,
        cluster_field=cluster_field,
        seed=seed
    )

    return clustered_examples, subset_df


def save_evaluation_files(
        clustered_examples: List[Dict],
        subset_df: pd.DataFrame,
        output_dir: Union[str, Path],
        clustered_filename: str = "teacher_with_intent_clusters.jsonl",
        eval_filename: str = "manual_eval_subset.csv",
        eval_columns: Optional[List[str]] = None
) -> tuple[Path, Path]:
    """
    Save clustered data and evaluation CSV to disk.

    Args:
        clustered_examples: Examples with cluster assignments
        subset_df: Subset DataFrame for evaluation
        output_dir: Directory to save files
        clustered_filename: Filename for clustered JSONL
        eval_filename: Filename for evaluation CSV
        eval_columns: Columns to include in eval CSV (None = use defaults)

    Returns:
        Tuple of (clustered_path, eval_csv_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save clustered data
    clustered_path = output_dir / clustered_filename
    save_jsonl(clustered_examples, clustered_path)

    # Prepare and save evaluation CSV
    if eval_columns is None:
        eval_columns = [
            "id", "intent_cluster", "intent", "command",
            "state", "actions", "rubric_correctness",
            "rubric_redundancy", "rubric_order", "rubric_notes"
        ]

    eval_df = prepare_manual_eval_csv(subset_df, keep_columns=eval_columns)
    eval_csv_path = output_dir / eval_filename
    eval_df.to_csv(eval_csv_path, index=False)

    return clustered_path, eval_csv_path