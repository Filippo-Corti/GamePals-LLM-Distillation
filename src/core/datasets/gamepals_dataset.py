from typing import TypeVar, Generic, Iterable, Type
import json

T = TypeVar('T')


class GamePalsDataset(Generic[T]):
    """
    GamePalsDataset is the abstract base class for all datasets
    """

    def __init__(self, items: Iterable[T] | None = None):
        self.items = list(items) if items else list()

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

    def append(self, item: T) -> None:
        self.items.append(item)

    def apply(self, transform: "GamePalsDatasetTransformer") -> "GamePalsDataset":
        return transform.transform(self)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(
                [item.model_dump() for item in self.items],
                f,
                indent=4
            )

    @staticmethod
    def load(path: str, cls: Type) -> "GamePalsDataset":
        with open(path, 'r') as f:
            items = json.load(f)
            x = GamePalsDataset()
            for item in items:
                x.append(cls(**item))
            return x