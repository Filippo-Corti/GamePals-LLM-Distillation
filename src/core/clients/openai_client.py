import json
import time
import math
from pathlib import Path
from typing import TypeVar, Generic, Callable, Optional
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI
from openai.types.responses import Response, ParsedResponse


class ProcessingMode(Enum):
    """Processing mode for dataset"""
    SEQUENTIAL = "Sequential"  # One-by-one API calls
    BATCH = "Batch"  # Batch API processing


I = TypeVar('I')
O = TypeVar('O')


def _pydantic_to_strict_schema(model: type[BaseModel]) -> dict:
    """
    Converts a Pydantic model to a strict JSON Schema for batch JSONL requests.

    The Responses API batch endpoint requires a raw JSON Schema in the request
    body, so we cannot rely on the SDK's .parse() abstraction here. This helper
    patches the generated schema so it satisfies OpenAI's strict mode:
      - every object gets  additionalProperties: false
      - every object's properties are all listed under required
    """
    def _make_strict(schema: dict) -> dict:
        if schema.get("type") == "object":
            props = schema.get("properties", {})
            schema["additionalProperties"] = False
            schema["required"] = list(props.keys())
            for prop_schema in props.values():
                _make_strict(prop_schema)
        for sub in schema.get("$defs", {}).values():
            _make_strict(sub)
        if "items" in schema:
            _make_strict(schema["items"])
        return schema

    return _make_strict(model.model_json_schema())


class OpenAIClient(Generic[I, O]):
    """
    A client for processing datasets with OpenAI API.

    Supports both sequential and batch processing modes, with optional
    structured outputs via a Pydantic BaseModel schema.
    """

    def __init__(
            self,
            model: str,
            mode: ProcessingMode,
            max_output_tokens: int = 1024,
            temperature: float = 1.0,
            reasoning_effort: str = 'none',
            tool_choice: str = 'auto',
            working_dir: Path = Path("./data/openai-client"),
            api_key: Optional[str] = None
    ):
        """
        Creates an OpenAIClient instance.

        :param model: OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini")
        :param mode: Processing mode (sequential or batch)
        :param max_output_tokens: Maximum tokens in model response (default: 1024)
        :param temperature: Sampling temperature (0-2) (default: 1.0)
        :param reasoning_effort: Reasoning effort for the model (default: none)
        :param tool_choice: Tool choice strategy - "auto", "none", "required", ... (default: "auto")
        :param working_dir: Directory for temporary files used by batch mode
        :param api_key: OpenAI API key (uses OPENAI_API_KEY env variable if not provided)
        """
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.mode = mode
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.working_dir = working_dir

    def process(
            self,
            dataset: list[I],
            system_prompt: str,
            tools: list[dict],
            format_input: Callable[[I], str],
            parse_output: Callable[[Response | ParsedResponse, str, float], O],
            get_id: Callable[[I, int], str] = lambda x, idx: x.get('id', f'item-{idx}'),
            batch_size: int = 1,
            request_delay: float = 0.0,
            output_schema: Optional[type[BaseModel]] = None,
    ) -> list[O]:
        """
        Processes a dataset through the OpenAI API.

        :param dataset: List of items to process
        :param system_prompt: System prompt for all requests
        :param tools: List of tool call signatures
        :param format_input: Function to convert each item to its prompt-ready representation
        :param parse_output: Function to convert each OpenAI Response to the output type.
            Also receives the input id and the latency (s) of the call.
            In sequential mode with output_schema set, receives a ParsedResponse —
            access the typed result via response.output_parsed.
            In batch mode with output_schema set, receives a standard Response —
            deserialise via output_schema.model_validate_json(response.output_text).
        :param get_id: Function that generates IDs for each item (default: "item-{index}")
        :param batch_size: Number of items per batch file (batch mode only, default: 1)
        :param request_delay: Seconds to wait between requests (sequential mode only, default: 0)
        :param output_schema: Optional Pydantic BaseModel class that activates structured
            outputs. The SDK handles schema serialisation automatically in sequential mode
            (.parse()); batch mode converts it to a strict JSON Schema for the JSONL body.
            When None (default), plain-text responses are returned as before.
        :return: List of parsed outputs in the same order as the input dataset
        """
        if not dataset:
            return []

        print(f"Processing mode: {self.mode.value}")
        print(f"Model: {self.model}")
        if tools:
            print(f"Tool calling enabled with {len(tools)} tool(s)")
        if output_schema is not None:
            print(f"Structured outputs enabled (schema: '{output_schema.__name__}')")

        match self.mode:
            case ProcessingMode.SEQUENTIAL:
                return self._process_sequential(
                    dataset=dataset,
                    system_prompt=system_prompt,
                    tools=tools,
                    format_input=format_input,
                    parse_output=parse_output,
                    get_id=get_id,
                    request_delay=request_delay,
                    output_schema=output_schema,
                )
            case ProcessingMode.BATCH:
                return self._process_batch(
                    dataset=dataset,
                    system_prompt=system_prompt,
                    tools=tools,
                    format_input=format_input,
                    parse_output=parse_output,
                    get_id=get_id,
                    batch_size=batch_size,
                    output_schema=output_schema,
                )

    def _process_sequential(
            self,
            dataset: list[I],
            system_prompt: str,
            tools: list[dict],
            format_input: Callable[[I], str],
            parse_output: Callable[[Response | ParsedResponse, str, float], O],
            get_id: Callable[[I, int], str] = lambda x, idx: x['id'] if 'id' in x else f'item-{idx}',
            request_delay: float = 0.0,
            output_schema: Optional[type[BaseModel]] = None,
    ) -> list:
        results = []
        total = len(dataset)

        print(f"Processing {total} items sequentially")

        for idx, item in enumerate(dataset):
            item_id = get_id(item, idx)
            print(f"[{idx + 1}/{total}] Processing item with id={item_id}...")

            request = {
                "model": self.model,
                "input": [{
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system_prompt}]
                }, {
                    "role": "user",
                    "content": [{"type": "input_text", "text": format_input(item)}]
                }],
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "reasoning": {"effort": self.reasoning_effort},
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }

            if tools:
                request["tool_choice"] = self.tool_choice
                request["tools"] = tools

            try:
                start_time = time.time()

                if output_schema is not None:
                    # .parse() handles schema serialisation and strict mode automatically.
                    # The text.format block is replaced by the text_format parameter.
                    del request["text"]
                    response = self.client.responses.parse(
                        **request,
                        text_format=output_schema,
                    )
                else:
                    response = self.client.responses.create(**request)

                latency = time.time() - start_time
                result = parse_output(response, item_id, latency)
            except Exception as e:
                print(e)
                result = None

            results.append(result)
            if request_delay > 0:
                time.sleep(request_delay)

        successful = sum(1 for r in results if r is not None)
        print(f"\nCompleted: {successful}/{total} successful")

        return results

    def _process_batch(
            self,
            dataset: list[I],
            system_prompt: str,
            tools: list[dict],
            format_input: Callable[[I], str],
            parse_output: Callable[[Response, str, float], O],
            get_id: Callable[[I, int], str] = lambda x, idx: x['id'] if 'id' in x else f'item-{idx}',
            batch_size: int = 1,
            output_schema: Optional[type[BaseModel]] = None,
    ) -> list:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        results = dict()
        total = len(dataset)
        num_batches = math.ceil(total / batch_size)

        print(f"Processing {total} items in {num_batches} batch(es)")

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total)
            batch_subset = dataset[start_idx:end_idx]

            print(f"\n{'=' * 60}")
            print(f"Batch {batch_num + 1}/{num_batches}")
            print(f"Items {start_idx} to {end_idx - 1} ({len(batch_subset)} total)")
            print(f"{'=' * 60}")

            batch_file = self._build_batch_file(
                batch_num=batch_num,
                dataset_subset=batch_subset,
                tools=tools,
                start_idx=start_idx,
                system_prompt=system_prompt,
                format_input=format_input,
                get_id=get_id,
                output_schema=output_schema,
            )

            start_time = time.time()
            batch_id = self._submit_batch(batch_file)
            status = self._wait_for_batch(batch_id)

            if status != "completed":
                raise RuntimeError(f"Batch {batch_id} ended with status: {status}")

            latency = time.time() - start_time
            batch_results = self._load_batch_results(
                batch_id=batch_id,
                parse_output=parse_output,
                latency=latency,
            )
            results.update(batch_results)

            print(f"Batch {batch_num + 1}/{num_batches} completed successfully!")

        # Return results in original order
        return [
            results.get(get_id(item, idx))
            for idx, item in enumerate(dataset)
        ]

    def _build_batch_file(
            self,
            batch_num: int,
            dataset_subset: list[I],
            tools: list[dict],
            start_idx: int,
            system_prompt: str,
            format_input: Callable[[I], str],
            get_id: Callable[[I, int], str],
            output_schema: Optional[type[BaseModel]] = None,
    ) -> Path:
        """Builds JSONL batch file for a subset of the dataset."""
        batch_file = self.working_dir / f"batch-{batch_num:04d}.jsonl"

        # .parse() is not available for batch JSONL — build the strict schema manually.
        text_format = (
            {
                "type": "json_schema",
                "name": output_schema.__name__,
                "schema": _pydantic_to_strict_schema(output_schema),
                "strict": True,
            }
            if output_schema is not None
            else {"type": "text"}
        )

        with open(batch_file, "w", encoding="utf-8") as f:
            for local_idx, item in enumerate(dataset_subset):
                global_idx = start_idx + local_idx
                custom_id = get_id(item, global_idx)

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": self.model,
                        "max_output_tokens": self.max_output_tokens,
                        "temperature": self.temperature,
                        "reasoning": {"effort": self.reasoning_effort},
                        "tools": tools,
                        "text": {"format": text_format},
                        "input": [
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": format_input(item)},
                        ],
                    },
                }

                f.write(json.dumps(request) + "\n")

        return batch_file

    def _submit_batch(self, batch_file: Path) -> str:
        """Uploads batch file and creates batch job."""
        with open(batch_file, "rb") as f:
            uploaded_file = self.client.files.create(file=f, purpose="batch")

        batch = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )

        return batch.id

    def _wait_for_batch(self, batch_id: str) -> str:
        """Waits for batch to complete, polling at regular intervals."""
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status

            total = batch.request_counts.total
            completed = batch.request_counts.completed
            failed = batch.request_counts.failed

            print(f"Status: {status} | Progress: {completed}/{total} | Failed: {failed}")

            if status in ("completed", "failed", "cancelling", "cancelled", "expired"):
                return status

            time.sleep(60)

    def _load_batch_results(
            self,
            batch_id: str,
            parse_output: Callable[[Response, str, float], O],
            latency: float,
    ) -> dict[str, O]:
        """Parses results from a completed batch."""
        batch = self.client.batches.retrieve(batch_id)

        content = self.client.files.content(batch.output_file_id)

        results = {}
        for line in content.iter_lines():
            record = json.loads(line)
            custom_id = record["custom_id"]
            response_dict = record["response"]["body"]

            try:
                response = Response.model_validate(response_dict)
                results[custom_id] = parse_output(response, custom_id, latency)
            except Exception as e:
                print(e)
                results[custom_id] = None

        return results

    def recover_batch_results(
            self,
            batch_id: str,
            parse_output: Callable[[Response, str, float], O],
            latency: float = 0.0,
    ) -> dict[str, O]:
        """
        Recovers results from a previously completed batch.

        :param batch_id: The ID of the batch to recover
        :param parse_output: Function to convert each OpenAI Response to the output type
        :param latency: Latency value to pass to parse_output (default: 0.0)
        :return: Dictionary mapping custom_id to parsed results
        :raises RuntimeError: If batch is not completed or has no output file
        """
        print(f"Recovering batch results for batch_id: {batch_id}")

        batch = self.client.batches.retrieve(batch_id)
        status = batch.status

        print(f"Batch status: {status}")

        if status != "completed":
            raise RuntimeError(
                f"Cannot recover batch {batch_id}: status is '{status}', not 'completed'"
            )

        if not batch.output_file_id:
            raise RuntimeError(f"Batch {batch_id} has no output file available")

        total = batch.request_counts.total
        completed = batch.request_counts.completed
        failed = batch.request_counts.failed

        print(f"Batch info: {completed}/{total} completed | {failed} failed")

        results = self._load_batch_results(batch_id, parse_output, latency)

        print(f"Successfully recovered {len(results)} results")

        return results