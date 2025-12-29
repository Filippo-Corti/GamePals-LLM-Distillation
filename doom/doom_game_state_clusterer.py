from typing import Callable, Counter
from sklearn.cluster import AgglomerativeClustering
import numpy as np

from datasets import GamePalsDataset
from .doom_game_state import DoomGameState, MonsterType, WeaponName, AimedAtType
from knowledge.dataset_clusterer import DatasetClusterer


class DoomGameStateClusterer(DatasetClusterer):
    """
    A DatasetClusterer specialized for doom game states
    """

    def __init__(self):
        """Creates a DoomGameStateClusterer"""
        super().__init__(
            to_features=DoomGameStateClusterer.to_features
        )

    @staticmethod
    def one_hot(value: str, vocab: list[str]) -> list[float]:
        vec = [0.0] * len(vocab)
        if value in vocab:
            vec[vocab.index(value)] = 1.0
        return vec

    @staticmethod
    def bucket_distance(d: float) -> float:
        if d < 256:
            return 0.0
        elif d < 768:
            return 0.5
        else:
            return 1.0

    @staticmethod
    def ammo_status(ammo: int) -> float:
        if ammo == 0:
            return 0.0
        elif ammo < 10:
            return 0.33
        elif ammo < 40:
            return 0.66
        else:
            return 1.0

    @staticmethod
    def to_features(state: DoomGameState) -> np.ndarray:
        features = []

        # --- Number of monsters ---
        features.append(float(len(state.MONSTERS)))

        # --- Closest enemy distance ---
        if state.MONSTERS:
            closest = min(m.distance for m in state.MONSTERS)
            features.append(DoomGameStateClusterer.bucket_distance(closest))
        else:
            features.append(1.0)

        # --- Most common enemy type (one-hot) ---
        if state.MONSTERS:
            types = [m.monsterType for m in state.MONSTERS]
            common_type = Counter(types).most_common(1)[0][0]
        else:
            common_type = None

        features.extend(DoomGameStateClusterer.one_hot(common_type, list(MonsterType)))

        # --- Ammo status ---
        slot = state.INVENTORY.inventorySlots[state.INVENTORY.currentSlot]
        features.append(DoomGameStateClusterer.ammo_status(slot.ammoCount))

        # --- Weapon name (one-hot) ---
        features.extend(DoomGameStateClusterer.one_hot(slot.weaponName.lower(), list(WeaponName)))

        # --- Is aiming at interactable ---
        features.append(float(state.AIMED_AT.interactable))

        # --- Aimed-at type (one-hot) ---
        aimed_type = state.AIMED_AT.entityType.lower() if state.AIMED_AT.entityType else "none"
        features.extend(DoomGameStateClusterer.one_hot(aimed_type, list(AimedAtType)))

        return np.array(features, dtype=np.float32)
