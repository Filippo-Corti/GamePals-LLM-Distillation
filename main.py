import os

from datasets import GamePalsDataset
from doom.doom_game_state import DoomGameState
from doom.doom_game_state_clusterer import DoomGameStateClusterer
from doom.doom_game_state_filterer import DoomGameStateFilterer
from knowledge.dataset_clusterer import DatasetClusterer

gamestates = [
    DoomGameState.model_validate_json(line[15:])
    for filepath in os.scandir("data/gamelogs")
    for line in open(filepath)
    if line.startswith("[GS] GAMESTATE ")
]
dataset = GamePalsDataset(gamestates)

print(len(dataset))

dataset = dataset.apply(
    DoomGameStateFilterer()
)

print(len(dataset))

dataset = dataset.apply(
    DoomGameStateClusterer()
)

print(len(dataset))


