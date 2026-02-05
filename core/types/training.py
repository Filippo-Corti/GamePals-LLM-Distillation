import json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class TrainingEntryMetrics:
    """Metrics for a training example. Useful for sampling"""
    explicitness: float
    atomicity: float
    contextuality: float
    cluster_id: int

@dataclass
class TrainingEntry:
    id: str
    game_state: str
    user_command: str
    expected_actions: str
    metrics: TrainingEntryMetrics