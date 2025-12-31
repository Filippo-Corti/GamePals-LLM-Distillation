from typing import Iterable
import random
import numpy as np

from core.knowledge.dataset_perturbator import DatasetPerturbator
from doom.doom_game_state import DoomGameState, AimedAtType, MonsterModel


class DoomGameStatePerturbator(DatasetPerturbator):
    """
    A DatasetPerturbator specialized for doom game states
    """
    N_MONSTER_PERTURBATIONS = 2
    N_AMMO_PERTURBATIONS = 2
    DROP_PROBABILITY = 0.3

    def __init__(self):
        """Creates a DoomGameStatePerturbator"""
        super().__init__(
            perturbate=self.perturbate
        )

    @staticmethod
    def perturbate_number(x: float, p: float = 0.5, delta: float = 0.1) -> float:
        if random.random() > p:
            return x
        scale = max(abs(x) * delta, 1E-3)
        noise = np.random.normal(loc=0.0, scale=scale)
        return x + noise

    @staticmethod
    def perturbate(state: DoomGameState) -> Iterable[DoomGameState]:
        # Tweak distance and position of each monster
        if state.AIMED_AT.entityType != AimedAtType.MONSTER:
            for i in range(DoomGameStatePerturbator.N_MONSTER_PERTURBATIONS):
                monsters = [
                    m.model_copy(
                        update=dict(
                            distance=max(DoomGameStatePerturbator.perturbate_number(m.distance, p=0.7, delta=0.1), 25),
                            relativeAngle=DoomGameStatePerturbator.perturbate_number(m.relativeAngle, p=0.7, delta=0.1),
                            relativePitch=DoomGameStatePerturbator.perturbate_number(m.relativePitch, p=0.7, delta=0.1),
                        ),
                        deep=True
                    )
                    for m in state.MONSTERS
                    if random.random() > DoomGameStatePerturbator.DROP_PROBABILITY
                ]
                yield state.model_copy(
                    update=dict(
                        MONSTERS=monsters
                    ),
                    deep=True
                )

        # Tweak inventory ammunition count
        for i in range(DoomGameStatePerturbator.N_AMMO_PERTURBATIONS):
            slots = [
                s.model_copy(
                    update=dict(
                        ammoCount=max(
                            int(round(DoomGameStatePerturbator.perturbate_number(s.ammoCount, p=0.7, delta=0.3))
                                ), 0),
                        canUse=s.canUse if not s.canUse else random.random() > DoomGameStatePerturbator.DROP_PROBABILITY
                    ),
                    deep=True
                )
                for s in state.INVENTORY.inventorySlots
            ]
            yield state.model_copy(
                update=dict(
                    INVENTORY=state.INVENTORY.model_copy(
                        update=dict(
                            inventorySlots=slots
                        ),
                        deep=True
                    )
                ),
                deep=True
            )
