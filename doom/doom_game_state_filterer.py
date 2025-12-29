from datasets import GamePalsDatasetTransformer, GamePalsDataset
from .doom_game_state import DoomGameState


class DoomGameStateFilterer(GamePalsDatasetTransformer):
    """
    A GamePalsDatasetTransformers specialized to filter out Doom Game States
    that are not considered relevant for the task of Commanding an LLM in Shared Control
    """

    def __init__(self):
        """Creates a DoomGameStateFilterer"""
        super().__init__()

    def transform(self, x: GamePalsDataset[DoomGameState]) -> GamePalsDataset[DoomGameState]:
        """
        Filters out Game States any game state for which neither of the conditions apply:
        * there are monsters
        * the player is aiming at an interactable object

        :param x: a gamepals dataset
        :return: the new gamepals dataset
        """
        new_x: list[DoomGameState] = list()
        for state in x:
            if len(state.MONSTERS) > 0:
                new_x.append(state)
            elif state.AIMED_AT.interactable:
                new_x.append(state)
        return GamePalsDataset[DoomGameState](new_x)
