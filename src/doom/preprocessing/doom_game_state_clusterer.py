from typing import Counter
import numpy as np

from src.doom.utils.doom_game_state import DoomGameState, MonsterType, WeaponName, AimedAtType
from src.core.knowledge.dataset_clusterer import DatasetClusterer


class DoomGameStateClusterer(DatasetClusterer):
    """
    A DatasetClusterer specialized for doom game states
    """

    def __init__(self):
        """Creates a DoomGameStateClusterer"""
        super().__init__(
            to_features=self.to_features
        )

    # Apply clustering to keep unique situations

    @staticmethod
    def bucket_distance(d: float) -> float:
        if d < 256: return 0.0
        if d < 768: return 0.5
        return 1.0

    @staticmethod
    def one_hot(value: str, vocab: list[str]) -> list[float]:
        vec = [0.0] * len(vocab)
        if value in vocab:
            vec[vocab.index(value)] = 1.0
        return vec

    @staticmethod
    def ammo_status(ammo: int) -> float:
        if ammo == 0: return 0.0
        if ammo < 10: return 0.33
        if ammo < 40: return 0.66
        return 1.0

    @classmethod
    def to_feature_vector(cls, gs: DoomGameState) -> np.ndarray:
        features = list()
        features.append(float(len(gs.MONSTERS)))  # Feature #1: Number of Monsters

        if gs.MONSTERS:
            closest = min(m.distance for m in gs.MONSTERS)
            features.append(cls.bucket_distance(closest))  # Feature #2: Distance to the closest Monster
            types = [m.monsterType for m in gs.MONSTERS]
            common_type = Counter(types).most_common(1)[0][0]
            features.extend(cls.one_hot(common_type, list(MonsterType)))  # Feature #3: OHE Most common Enemy Type
        else:
            features.append(1.0)
            features.extend([0.0] * len(MonsterType))

        slot = gs.INVENTORY.inventorySlots[gs.INVENTORY.currentSlot]
        features.append(cls.ammo_status(slot.ammoCount))  # Feature #4: Ammunition count

        features.extend(cls.one_hot(slot.weaponName.lower(), list(WeaponName)))  # Feature #5: OHE Current Weapon

        features.append(float(gs.AIMED_AT.interactable))  # Feature #6: is aiming at interactable

        aimed_type = gs.AIMED_AT.entityType.lower() if gs.AIMED_AT.entityType else "none"
        features.extend(cls.one_hot(aimed_type, list(AimedAtType)))  # Feature #6: OHE aimed at entity type
        return np.array(features, dtype=np.float32)