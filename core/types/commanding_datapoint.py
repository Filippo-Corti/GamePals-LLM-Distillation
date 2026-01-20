from dataclasses import dataclass

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
