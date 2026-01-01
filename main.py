import os
import dotenv

from core.datasets import GamePalsDataset
from doom.kd.doom_teacher import DoomTeacher, DoomTeacherOptions
from doom.preprocessing.doom_game_state_clusterer import DoomGameStateClusterer
from doom.preprocessing.doom_game_state_filterer import DoomGameStateFilterer
from doom.preprocessing.doom_game_state_perturbator import DoomGameStatePerturbator
from doom.utils.doom_game_state import DoomGameState

dotenv.load_dotenv()

#
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

teacher = DoomTeacher(
    game_states=dataset,
    options=DoomTeacherOptions(
        prompt_data_filepath='prompts/doom-prompt-data.json',
        open_ai_model='gpt-5.1',
        user_commands_batch_input_filepath='data/batches/user-commands-input.jsonl',
        user_commands_batch_output_filepath='data/batches/user-commands-output.json'
    )
)

generation_prompt = open('prompts/command-generation-template.md', 'r').read()
teacher.generate_user_commands(generation_prompt)
