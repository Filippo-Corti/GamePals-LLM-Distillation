import json
from dataclasses import dataclass

from core.datasets import GamePalsDataset
from core.knowledge.gamepals_teacher import GamePalsTeacher

@dataclass
class DoomTeacherOptions:
    """
    The options for a DoomTeacher instantiation

    :ivar str prompt_data_filepath: the path to the file containing doom-specific parts of the prompt
    :ivar str open_ai_model: the name of the model to use as teacher
    """
    prompt_data_filepath: str
    open_ai_model: str

class DoomTeacher(GamePalsTeacher):
    """
    DoomTeacher is a Doom-specific implementation for a Knowledge Distillation Teacher.
    """

    def __init__(self, game_states: GamePalsDataset, options: DoomTeacherOptions):
        """
        Creates a DoomTeacher.

        :param game_states: the dataset of game states from which the teacher should elicit their knowledge
        :param DoomTeacherOptions options: the options for the instantiation of the teacher
        """
        super().__init__(game_states)
        self.options = options


    def generate_user_commands(self, prompt: str):
        """
        Generates user commands for the dataset of game states,
        using the OpenAI API to access a state-of-the-art black-box LLM.

        :param prompt: the knowledge-elicitation prompt
        """
        prompt_data = json.load(open(self.options.prompt_data_filepath, "r"))

        full_prompt = prompt
        for tag, value in prompt_data.items():
            if type(value) is list:
                value = "\n".join(value)
            full_prompt = full_prompt.replace(f"<{tag}>", value)

        print(full_prompt)

    def generate_labels(self, prompt: str):
        pass