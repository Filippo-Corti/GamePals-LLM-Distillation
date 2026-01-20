import json
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel


class GameState(BaseModel):
    # Game State fields go here as BaseModels

    def to_prompt_ready(self) -> str:
        """:return: the string representation of the game state"""
        return ""


T = TypeVar("T", bound=GameState)


@dataclass
class GameStateEntry:
    id: str
    state: GameState

    def asdict(self) -> dict:
        return {
            "id": self.id,
            "state": self.state.model_dump(),
        }

    @staticmethod
    def fromdict(d: dict, cls: Type[T]) -> "GameStateEntry":
        return GameStateEntry(id=d['id'], state=cls(**d['state']))

    @staticmethod
    def save_states(x: list["GameStateEntry"], path: Path):
        with open(path, 'w') as f:
            json.dump([item.asdict() for item in x], f, indent=4)

    @staticmethod
    def load_states(path: Path, cls: Type[T]) -> list["GameStateEntry"]:
        with open(path, 'r') as f:
            items = json.load(f)
            x = list()
            for item in items:
                x.append(GameStateEntry.fromdict(item, cls))
            return x
