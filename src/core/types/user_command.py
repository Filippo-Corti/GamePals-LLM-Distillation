import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class UserCommand:
    command: str
    intent: str
    explicitness: float
    atomicity: str
    contextuality: str

    def asdict(self) -> dict:
        return asdict(self)

    @staticmethod
    def fromdict(d: dict) -> "UserCommand":
        return UserCommand(**d)

@dataclass
class UserCommandEntry:
    id: str
    state_id: str
    latency: float
    command: UserCommand

    def asdict(self) -> dict:
        return asdict(self)

    @staticmethod
    def fromdict(d: dict) -> "UserCommandEntry":
        return UserCommandEntry(
            id=d['id'],
            state_id=d['state_id'],
            latency=d['latency'],
            command=UserCommand.fromdict(d['command']),
        )

    @staticmethod
    def save_commands(x: list["UserCommandEntry"], path: Path):
        with open(path, 'w') as f:
            json.dump([item.asdict() for item in x], f, indent=4)

    @staticmethod
    def load_commands(path: Path) -> list["UserCommandEntry"]:
        with open(path, 'r') as f:
            items = json.load(f)
            x = list()
            for item in items:
                x.append(UserCommandEntry.fromdict(item))
            return x
