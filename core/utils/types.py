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


@dataclass
class LLMCommandingDataPoint:
    input_id: str
    game_state: str
    command: str
    command_intent: str
    command_explicitness: float
    command_atomicity: float
    command_contextuality: float
    game_actions: str
    latency: float
    reason_if_failed: str = ''
    cluster_id: int | None = None
    selected_for_labelling: bool = False


@dataclass
class LLMCommandingLabelledDataPoint(LLMCommandingDataPoint):
    exact_match: float | None = None
    edit_distance: float | None = None
    action_full_correct: int = 0
    action_unnecessary: int = 0
    action_imprecise_sequentiality: int = 0
    action_imprecise_parameters: int = 0
    action_harming_sequentiality: int = 0
    action_harming_parameters: int = 0
    action_missing: int = 0
    action_harming: int = 0
    action_wrong_syntax: int = 0
    label: str | None = None
