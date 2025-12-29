from typing import TypeVar, Generic, Iterable

T = TypeVar('T')


class GamePalsDataset(Generic[T]):
    """
    GamePalsDataset is the abstract base class for datasets TODO
    """

    def __init__(self, items: Iterable[T]):
        self.items = list(items)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

    def apply(self, transform: "GamepalsDatasetTransformer") -> "GamePalsDataset":
        return transform.transform(self)
