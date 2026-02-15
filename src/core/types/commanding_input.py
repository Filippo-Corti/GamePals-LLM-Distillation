import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Type

from .game_state import GameStateEntry, GameState
from .user_command import UserCommandEntry, UserCommand


@dataclass
class LLMCommandingInput:
    id: str
    game_state: GameStateEntry
    user_command: UserCommandEntry
    cluster_id: int = -1
    selected_for_labelling: bool = False

    def asdict(self) -> dict:
        return {
            'id': self.id,
            'game_state': GameStateEntry.asdict(self.game_state),
            'user_command': asdict(self.user_command),
            'cluster_id': self.cluster_id,
            'selected_for_labelling': self.selected_for_labelling,
        }

    @staticmethod
    def fromdict(d: dict, cls: Type[GameState]) -> 'LLMCommandingInput':
        return LLMCommandingInput(
            id=d['id'],
            game_state=GameStateEntry.fromdict(d['game_state'], cls),
            user_command=UserCommandEntry.fromdict(d['user_command']),
            cluster_id=d['cluster_id'],
            selected_for_labelling=d['selected_for_labelling'],
        )

    @staticmethod
    def save_inputs(x: list["LLMCommandingInput"], path: Path):
        with open(path, 'w') as f:
            json.dump([item.asdict() for item in x], f, indent=4)

    @staticmethod
    def load_inputs(path: Path, gstype: Type[GameState]) -> list["LLMCommandingInput"]:
        with open(path, 'r') as f:
            items = json.load(f)
            x = list()
            for item in items:
                x.append(LLMCommandingInput.fromdict(item, gstype))
            return x

