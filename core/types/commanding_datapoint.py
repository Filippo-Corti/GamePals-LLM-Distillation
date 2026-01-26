from dataclasses import dataclass
from enum import Enum


class DataPointLabel(str, Enum):
    FULLY_CORRECT = "FC"
    CORRECT_BUT_NOT_OPTIMAL = "CNO"
    OPERATIONALLY_WRONG = "OW"
    CONCEPTUALLY_WRONG = "CW"
    SYNTACTICALLY_WRONG = "SW"


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
    action_type_f1: float | None = None
    action_type_f1_lcs: float | None = None
    action_full_correct: int = 0
    action_unnecessary: int = 0
    action_imprecise_sequentiality: int = 0
    action_imprecise_parameters: int = 0
    action_harming_sequentiality: int = 0
    action_harming_parameters: int = 0
    action_missing: int = 0
    action_harming: int = 0
    action_wrong_syntax: int = 0
    label: DataPointLabel | None = None

    def infer_label(self):
        """Sets self.label based on the actions classification"""
        if self.action_wrong_syntax > 0:
            self.label = DataPointLabel.SYNTACTICALLY_WRONG
        elif self.action_missing > 0 or self.action_harming > 0:
            self.label = DataPointLabel.CONCEPTUALLY_WRONG
        elif self.action_harming_parameters > 0 or self.action_harming_sequentiality > 0:
            self.label = DataPointLabel.OPERATIONALLY_WRONG
        elif self.action_unnecessary > 0 or self.action_imprecise_parameters > 0 or self.action_imprecise_sequentiality > 0:
            self.label = DataPointLabel.CORRECT_BUT_NOT_OPTIMAL
        else:
            self.label = DataPointLabel.FULLY_CORRECT