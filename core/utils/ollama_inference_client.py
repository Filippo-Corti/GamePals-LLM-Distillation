import time
from typing import TypeVar, Generic, Callable, Optional
from tqdm import tqdm
from ollama import Client

I = TypeVar("I")
O = TypeVar("O")


class OllamaInferenceClient(Generic[I, O]):
    """
    Simple, fast inference client using the official Ollama Python package.
    This replaces manual HTTP requests with proper API calls.
    """

    def __init__(
        self,
        model: str,
        max_output_tokens: int = 512,
        temperature: float = 1.0,
    ):
        """
        Initialize Ollama client.

        :param model: Ollama model name (e.g., "llama3.1:8b", "my-model:latest")
        :param max_output_tokens: Max tokens to generate
        :param temperature: Sampling temperature
        """
        self.model_name = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        # Initialize Ollama package client
        self.client = Client()
        print(f"🚀 Initialized OllamaInferenceClient for {self.model_name}")

    def process(
        self,
        dataset: list[I],
        system_prompt: str,
        format_input: Callable[[I], str],
        parse_output: Callable[[str, str, float], O],
        get_id: Callable[[I, int], str] = lambda x, idx: x.get("id", f"item-{idx}"),
    ) -> list[O]:
        """
        Process a dataset sequentially using Ollama.

        :param dataset: List of items
        :param system_prompt: System prompt (can include tool definitions)
        :param format_input: Convert each item to user message
        :param parse_output: Parse model output
        :param get_id: Generate IDs for items
        :return: List of parsed outputs
        """
        if not dataset:
            return []

        results = []
        total = len(dataset)
        print(f"🔄 Processing {total} items sequentially")

        for idx in tqdm(range(total), desc="Processing items"):
            item = dataset[idx]
            item_id = get_id(item, idx)
            user_message = format_input(item)

            try:
                output, latency = self._generate(
                    system_prompt=system_prompt, user_message=user_message
                )
                result = parse_output(output, item_id, latency)
            except Exception as e:
                print(f"\n❌ Error processing item {item_id}: {e}")
                result = None

            results.append(result)

        successful = sum(1 for r in results if r is not None)
        print(f"\n✅ Completed: {successful}/{total} successful")
        return results

    def _generate(self, system_prompt: str, user_message: str) -> tuple[str, float]:
        """
        Generate a response using the Ollama package.

        :param system_prompt: system prompt string
        :param user_message: user message string
        :return: tuple of (output string, latency in seconds)
        """
        start_time = time.time()

        # Ollama package chat call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=False,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            },
        )

        latency = time.time() - start_time
        return response.content.strip(), latency

    def list_models(self) -> list[dict]:
        """List available Ollama models."""
        models = self.client.list_models()
        print("\n📚 Available Ollama models:")
        for model in models:
            print(f"  - {model['name']}")
        return models

    def unload_model(self):
        """Explicitly stop the model (optional)."""
        try:
            self.client.stop_model(self.model_name)
            print(f"✓ Model {self.model_name} stopped")
        except Exception as e:
            print(f"⚠️ Could not stop model {self.model_name}: {e}")
