from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar('T')

@dataclass
class GameStateEntry(Generic[T]):
    id: str
    state: T

@dataclass
class UserCommand:
    command: str
    intent: str
    explicitness: float
    atomicity: str
    contextuality: str


@dataclass
class UserCommandEntry:
    id: str
    state_id: str
    latency: float
    command: UserCommand


@dataclass
class LLMCommandingInput:
    id: str
    game_state: GameStateEntry
    user_command: UserCommandEntry


@dataclass
class GameAction:
    action: str  # TODO: make enum?
    value: float
    duration: float
    blocking: bool


@dataclass
class LLMCommandingOutput:
    input_id: str
    actions: list[GameAction] | None
    reason: str | None
    latency: float

