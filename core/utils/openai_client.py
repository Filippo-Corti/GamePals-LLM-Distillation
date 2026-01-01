import json
import time
import math
from pathlib import Path
from typing import TypeVar, Generic, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI


class ProcessingMode(Enum):
    """Processing mode for dataset."""
    SEQUENTIAL = "sequential"  # One-by-one API calls
    BATCH = "batch"  # Batch API processing


class BatchStatus(Enum):
    """Status of a batch operation."""
    PENDING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class OpenAIClient:
    """
    A client for processing datasets with OpenAI API.

    Supports both sequential and batch processing modes.
    """

    def __init__(
            self,
            model: str,
            mode: ProcessingMode,
            max_output_tokens: int = 1024,
            temperature: float = 1.0,
            working_dir: Path = Path("./data/openai-client"),
            api_key: Optional[str] = None
    ):
        """
        Creates an OpenAIClient instance.

        :param model: OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        :param mode: Processing mode (sequential or batch)
        :param max_output_tokens: Maximum tokens in model response
        :param temperature: Sampling temperature (0-2)
        :param working_dir: Directory for temporary files
        :param api_key: OpenAI API key (uses environment variable if not provided)
        """
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.mode = mode
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def process(
            self,
            dataset: list[str],
            system_prompt: str,
            batch_size: int = 1,
            request_delay: float = 0.0
    ) -> list:
        if not dataset:
            return []

        print(f"Processing mode: {self.mode.value}")
        print(f"Model: {self.model}")

        match self.mode:
            case ProcessingMode.SEQUENTIAL:
                return self._process_sequential(
                    dataset,
                    system_prompt,
                    request_delay
                )
            case ProcessingMode.BATCH:
                return self._process_batch(
                    dataset,
                    system_prompt,
                    batch_size
                )

    def _process_sequential(
            self,
            dataset: list[str],
            system_prompt: str,
            request_delay: float = 0.0
    ) -> list:
        results = []
        total = len(dataset)

        print(f"Processing {total} items sequentially")

        for idx, item in enumerate(dataset):
            item_id = f'item-{idx}'

            print(f"[{idx + 1}/{total}] Processing {item_id}...", end=" ")

            try:
                # Make API call
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "developer",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": '' # TODO
                        }
                    ],
                    reasoning={"effort": "low"},
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                )

                result = '\n'.join([
                    out.content.text
                    for out in response.output
                    if out.type == 'message' and out.content.type == 'output_text'
                ])
            except Exception as e:
                print(e)
                result = None

            results.append(result)

            if request_delay > 0:
                time.sleep(request_delay)

        successful = sum(1 for r in results if r.success)
        print(f"\nCompleted: {successful}/{total} successful")

        return results

    def _process_batch(
            self,
            dataset: list[str],
            system_prompt: str,
            batch_size: int = 1
    ) -> list:
        num_batches = math.ceil(len(dataset) / batch_size)

        print(f"Processing {len(dataset)} items in {num_batches} batch(es)")

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(dataset))

            batch_subset = dataset[start_idx:end_idx]

            print(f"\n{'=' * 60}")
            print(f"Batch {batch_num + 1}/{num_batches}")
            print(f"Items {start_idx} to {end_idx - 1} ({len(batch_subset)} total)")
            print(f"{'=' * 60}")

            # Build and submit batch
            batch_file = self._build_batch_file(
                batch_num=batch_num,
                dataset_subset=batch_subset,
                start_idx=start_idx,
                system_prompt=system_prompt,
            )

            print("Submitting batch...")
            batch_id = self._submit_batch(batch_file)
            print(f"Batch ID: {batch_id}")

            # Wait for completion
            print("Waiting for batch to complete...")
            status = self._wait_for_batch(batch_id)

            if status != BatchStatus.COMPLETED:
                raise RuntimeError(
                    f"Batch {batch_id} ended with status: {status.value}"
                )

            # Load and parse results
            print("Loading results...")
            batch_results = self._load_batch_results(
                batch_id=batch_id,
                dataset_subset=batch_subset,
                start_idx=start_idx,
            )
            all_results.update(batch_results)

            print(f"Batch {batch_num + 1}/{num_batches} completed successfully!")

            # Return results in original order
        return self._order_results(all_results, len(dataset))

    def _build_batch_file(
            self,
            batch_num: int,
            dataset_subset: list[str],
            start_idx: int,
            system_prompt: str,
    ) -> Path:
        """Build JSONL batch file for a subset of the dataset."""
        batch_file = self.working_dir / f"batch-{batch_num:04d}.jsonl"

        with open(batch_file, "w", encoding="utf-8") as f:
            for local_idx, item in enumerate(dataset_subset):
                global_idx = start_idx + local_idx
                custom_id = f'item-{global_idx}'

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.model,
                        "max_output_tokens": self.max_output_tokens,
                        "temperature": self.temperature,  # TODO: verify if ignored or used
                        "input": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": '' # TODO (game state to prompt ready)
                            }
                        ]
                    }
                }

                f.write(json.dumps(request) + "\n")

        return batch_file

    def _submit_batch(self, batch_file: Path) -> str:
        """Upload batch file and create batch job."""
        with open(batch_file, "rb") as f:
            uploaded_file = self.client.files.create(file=f, purpose="batch")

        batch = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/responses",
            completion_window='24h',
        )

        return batch.id

    def _wait_for_batch(self, batch_id: str) -> BatchStatus:
        """Wait for batch to complete, polling at regular intervals."""
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = BatchStatus(batch.status)

            total = batch.request_counts.total
            completed = batch.request_counts.completed
            failed = batch.request_counts.failed

            print(
                f"Status: {status.value} | "
                f"Progress: {completed}/{total} | "
                f"Failed: {failed}"
            )

            if status in (
                    BatchStatus.COMPLETED,
                    BatchStatus.FAILED,
                    BatchStatus.CANCELLED,
                    BatchStatus.EXPIRED,
            ):
                return status

            time.sleep(60)

    def _load_batch_results(
            self,
            batch_id: str,
            dataset_subset: list[str],
            start_idx: int,
    ) -> dict[str, str]:
        """Load and parse results from a completed batch."""
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch_id} not completed: {batch.status}")

        output_file_id = batch.output_file_id
        content = self.client.files.content(output_file_id)

        results = {}
        for line in content.iter_lines():
            record = json.loads(line)
            custom_id = record["custom_id"]

            # Extract item index from custom_id
            try:
                output_text = ""
                for item in record["response"]["body"]["output"]:
                    for content_item in item["content"]:
                        if content_item["type"] == "output_text":
                            output_text += content_item["text"]

                results[custom_id] = output_text

            except Exception as e:
                results[custom_id] = None

        return results

    def _order_results(
        self,
        results: dict[str, str],
        total_items: int,
    ) -> list[str]:
        """Order results to match original dataset order."""
        ordered = []
        for i in range(total_items):
            item_id = f"item-{i}"
            if item_id in results:
                ordered.append(results[item_id])
            else:
                ordered.append(None)
        return ordered