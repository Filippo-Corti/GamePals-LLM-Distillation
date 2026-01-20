import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LLMCommandingOutput:
    input_id: str
    actions: str
    reason: str | None
    latency: float

    def asdict(self) -> dict:
        return asdict(self)

    @staticmethod
    def fromdict(d: dict) -> 'LLMCommandingOutput':
        return LLMCommandingOutput(**d)

    @staticmethod
    def save_outputs(x: list["LLMCommandingOutput"], path: Path):
        with open(path, 'w') as f:
            json.dump([item.asdict() for item in x], f, indent=4)

    @staticmethod
    def load_outputs(path: Path) -> list["LLMCommandingOutput"]:
        with open(path, 'r') as f:
            items = json.load(f)
            x = list()
            for item in items:
                x.append(LLMCommandingOutput.fromdict(item))
            return x
