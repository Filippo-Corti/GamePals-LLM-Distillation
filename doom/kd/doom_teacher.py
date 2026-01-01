import dataclasses
import json
import time

from openai import OpenAI
from dataclasses import dataclass

from core.datasets import GamePalsDataset
from core.knowledge.gamepals_teacher import GamePalsTeacher
from core.knowledge.utils import UserCommandInfo
from doom.utils.doom_game_state import DoomGameState


@dataclass
class DoomTeacherOptions:
    """
    The options for a DoomTeacher instantiation

    :ivar str prompt_data_filepath: the path to the file containing doom-specific parts of the prompt
    :ivar str open_ai_model: the name of the model to use as teacher
    TODO
    """
    prompt_data_filepath: str
    open_ai_model: str
    user_commands_batch_input_filepath: str
    user_commands_batch_output_filepath: str


class DoomTeacher(GamePalsTeacher):
    """
    DoomTeacher is a Doom-specific implementation for a Knowledge Distillation Teacher.
    """

    def __init__(self, game_states: GamePalsDataset[DoomGameState], options: DoomTeacherOptions):
        """
        Creates a DoomTeacher.

        :param game_states: the dataset of game states from which the teacher should elicit their knowledge
        :param DoomTeacherOptions options: the options for the instantiation of the teacher
        """
        super().__init__(game_states)
        self.options = options
        self.client = OpenAI()

    def generate_user_commands(self, prompt: str):
        """
        Generates user commands for the dataset of game states,
        using the OpenAI API to access a state-of-the-art black-box LLM.

        :param prompt: the knowledge-elicitation prompt
        """
        batch_file = self.options.user_commands_batch_input_filepath

        print("Building batch file...")
        self.build_batch_jsonl(prompt, batch_file)

        print("Submitting batch...")
        batch_id = self.submit_batch(batch_file)

        print(f"Batch submitted: {batch_id}")
        status = self.wait_for_batch(batch_id)

        if status != "completed":
            raise RuntimeError(f"Batch failed with status: {status}")

        print("Loading results...")
        self.load_batch_results(batch_id)

    def generate_labels(self, prompt: str):
        pass

    def build_full_prompt(self, base_prompt: str) -> str:
        prompt_data = json.load(open(self.options.prompt_data_filepath, "r"))

        full_prompt = base_prompt
        for tag, value in prompt_data.items():
            if isinstance(value, list):
                value = "\n".join(value)
            full_prompt = full_prompt.replace(f"<{tag}>", value)

        return full_prompt

    def build_batch_jsonl(self, base_prompt: str, output_path: str) -> None:
        full_prompt = self.build_full_prompt(base_prompt)

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, game_state in enumerate(self.game_states):
                if idx > 3: break  # TODO: remove for full batch
                request = {
                    "custom_id": f"state-{idx}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.options.open_ai_model,
                        "max_output_tokens": 256,
                        "temperature": 1.0, # TODO: verify if ignored or actually used
                        "input": [
                            {
                                "role": "system",
                                "content": full_prompt
                            },
                            {
                                "role": "user",
                                "content": game_state.to_prompt_ready()
                            }
                        ]
                    }
                }

                f.write(json.dumps(request) + "\n")

    def submit_batch(self, jsonl_path: str) -> str:
        uploaded_file = self.client.files.create(
            file=open(jsonl_path, "rb"),
            purpose="batch"
        )

        batch = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/responses",
            completion_window="24h"
        )

        return batch.id

    def wait_for_batch(self, batch_id: str, poll_interval: int = 60) -> str:
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            total = batch.request_counts.total
            completed = batch.request_counts.completed
            failed = batch.request_counts.failed

            print(f"[Batch {batch_id}] status = {status} - progress = {completed}/{total} (failed = {failed})")

            if status in ("completed", "failed", "cancelled"):
                return status

            time.sleep(poll_interval)

    def load_batch_results(self, batch_id: str) -> None:
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch ended with status: {batch.status}")

        output_file_id = batch.output_file_id
        content = self.client.files.content(output_file_id)

        # Parse jsonl output
        results = {}
        for line in content.iter_lines():
            record = json.loads(line)
            custom_id = record["custom_id"]

            output_text = ""
            for item in record["response"]["body"]["output"]:
                for content in item["content"]:
                    if content["type"] == "output_text":
                        output_text += content["text"]

            results[custom_id] = output_text.strip()

        # Reorder results
        self.user_commands = list()

        for i in range(len(self.game_states)):
            if f"state-{i}" in results:
                result = results[f"state-{i}"].split("\n")
                for line in result:
                    data = json.loads(line)
                    self.user_commands.append(
                        UserCommandInfo(
                            command=data["command"],
                            game_state_idx=i,
                            intent=data["intent"],
                            explicitness=data["explicitness"],
                            atomicity=data["atomicity"],
                            contextuality=data["contextuality"],
                        ))

        # Save to file
        with open(self.options.user_commands_batch_output_filepath, "w") as f:
            json.dump([dataclasses.asdict(command) for command in self.user_commands], f, indent=2)
