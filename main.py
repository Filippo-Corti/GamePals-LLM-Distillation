import os
import json

from core.datasets import GamePalsDataset
from doom.doom_game_state import DoomGameState
from doom.doom_game_state_clusterer import DoomGameStateClusterer
from doom.doom_game_state_filterer import DoomGameStateFilterer
from doom.doom_game_state_perturbator import DoomGameStatePerturbator

# gamestates = [
#     DoomGameState.model_validate_json(line[15:])
#     for filepath in os.scandir("data/gamelogs")
#     for line in open(filepath)
#     if line.startswith("[GS] GAMESTATE ")
# ]
# dataset = GamePalsDataset(gamestates)
#
# print(len(dataset))
# dataset.save("data/gamestates/original-gamestates.json")
#
# dataset = dataset.apply(
#     DoomGameStateFilterer()
# )
#
# print(len(dataset))
# dataset.save("data/gamestates/filtered-gamestates.json")
#
# dataset = dataset.apply(
#     DoomGameStateClusterer()
# )
#
# print(len(dataset))
# dataset.save("data/gamestates/clustered-gamestates.json")
#
# dataset = dataset.apply(
#     DoomGameStatePerturbator()
# )
#
# print(len(dataset))
# dataset.save("data/gamestates/perturbated-gamestates.json")

# Alternatively:
dataset = GamePalsDataset.load('data/gamestates/perturbated-gamestates.json', cls=DoomGameState)

print(len(dataset))
print(dataset[16])
