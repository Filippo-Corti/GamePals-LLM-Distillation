import dataclasses
import json
import math
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
    :ivar str user_commands_batch_input_filepath: the path for batch input files
    :ivar str user_commands_batch_output_filepath: the path for batch output files
    :ivar int max_tokens_per_batch: maximum tokens to enqueue per batch
    :ivar int estimated_tokens_per_request: estimated tokens per request for chunking
    """
    prompt_data_filepath: str
    open_ai_model: str
    user_commands_batch_input_filepath: str
    user_commands_batch_output_filepath: str
    max_tokens_per_batch: int
    estimated_tokens_per_request: int

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
        Splits into multiple batches if needed to stay under token limits.

        :param prompt: the knowledge-elicitation prompt
        """
        # Calculate how many batches we need
        requests_per_batch = self.options.max_tokens_per_batch // self.options.estimated_tokens_per_request
        num_batches = math.ceil(len(self.game_states) / requests_per_batch)

        print(f"Total game states: {len(self.game_states)}")
        print(f"Splitting into {num_batches} batch(es) with ~{requests_per_batch} requests each")

        all_batch_ids = []

        # Process batches sequentially: submit, wait for completion, then submit next
        for batch_num in range(num_batches):
            start_idx = batch_num * requests_per_batch
            end_idx = min((batch_num + 1) * requests_per_batch, len(self.game_states))

            batch_file = f"{self.options.user_commands_batch_input_filepath}.{batch_num}"

            print(f"\n=== Batch {batch_num + 1}/{num_batches} ===")
            print(f"Processing game states {start_idx} to {end_idx - 1}")
            print("Building batch file...")

            self.build_batch_jsonl(prompt, batch_file, start_idx, end_idx)

            print("Submitting batch...")
            batch_id = self.submit_batch(batch_file)
            all_batch_ids.append(batch_id)

            print(f"Batch submitted: {batch_id}")

            # Wait for THIS batch to complete before submitting the next one
            print(f"Waiting for batch {batch_num + 1}/{num_batches} to complete...")
            status = self.wait_for_batch(batch_id)

            if status != "completed":
                raise RuntimeError(f"Batch {batch_id} failed with status: {status}")

            print(f"Batch {batch_num + 1}/{num_batches} completed successfully!")

        # Load results from all batches
        print("\n=== Loading results from all batches ===")
        self.load_multiple_batch_results(all_batch_ids)

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

    def build_batch_jsonl(self, base_prompt: str, output_path: str, start_idx: int = 0, end_idx: int = None) -> None:
        """
        Build a batch JSONL file for a subset of game states.

        :param base_prompt: the base prompt to use
        :param output_path: where to save the JSONL file
        :param start_idx: starting index in game_states (inclusive)
        :param end_idx: ending index in game_states (exclusive), None for all
        """
        full_prompt = self.build_full_prompt(base_prompt)

        if end_idx is None:
            end_idx = len(self.game_states)

        with open(output_path, "w", encoding="utf-8") as f:
            for idx in range(start_idx, end_idx):
                game_state = self.game_states[idx]
                request = {
                    "custom_id": f"state-{idx}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.options.open_ai_model,
                        "max_output_tokens": 256,
                        "temperature": 1.0, # TODO: verify if ignored or used
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

    def load_multiple_batch_results(self, batch_ids: list[str]) -> None:
        """
        Load results from multiple batches and combine them.

        :param batch_ids: list of batch IDs to load results from
        """
        all_results = {}

        for i, batch_id in enumerate(batch_ids):
            print(f"Loading results from batch {i + 1}/{len(batch_ids)}: {batch_id}")
            batch_results = self.load_single_batch_results(batch_id)
            all_results.update(batch_results)

        self.user_commands = list()

        for i in range(len(self.game_states)):
            if f"state-{i}" in all_results:
                result = all_results[f"state-{i}"].split("\n")
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

        print(f"\nTotal user commands generated: {len(self.user_commands)}")

    def load_single_batch_results(self, batch_id: str) -> dict:
        """
        Load results from a single batch.

        :param batch_id: the batch ID to load
        :return: dictionary mapping custom_id to output text
        """
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status}")

        output_file_id = batch.output_file_id
        content = self.client.files.content(output_file_id)

        # Parse jsonl output
        results = {}
        for line in content.iter_lines():
            record = json.loads(line)
            custom_id = record["custom_id"]

            output_text = ""
            for item in record["response"]["body"]["output"]:
                for content_item in item["content"]:
                    if content_item["type"] == "output_text":
                        output_text += content_item["text"]

            results[custom_id] = output_text.strip()

        return results

