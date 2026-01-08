import json
import time

import torch
from pathlib import Path
from typing import TypeVar, Generic, Callable, Optional, Union, Any
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

I = TypeVar('I')
O = TypeVar('O')


class HuggingFaceClient(Generic[I, O]):
    """
    A client for processing datasets and fine-tuning models with Hugging Face.

    Supports local inference, fine-tuning, and tool calling.
    """

    def __init__(
            self,
            model: str,
            device: Optional[str] = None,
            max_output_tokens: int = 512,
            temperature: float = 1.0,
            working_dir: Path = Path("./data/huggingface"),
            load_in_8bit: bool = False,
            load_in_4bit: bool = False,
    ):
        """
        Creates a HuggingFaceClient instance.

        :param model: Hugging Face model name or path
        :param device: Device to use ('cuda', 'cpu', or None for auto)
        :param max_output_tokens: Maximum tokens to generate (default: 512)
        :param temperature: Sampling temperature (default: 1.0)
        :param working_dir: Directory to cache models
        :param load_in_8bit: Load model in 8-bit precision (default: False)
        :param load_in_4bit: Load model in 4-bit precision (default: False)
        """
        self.model_name = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.working_dir = working_dir
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        self.model = None
        self.tokenizer = None

        print(f"Initialized HuggingFaceClient for {self.model_name}")
        print(f"Device: {self.device}")

    def load_model(self, model_name: Optional[str] = None):
        """Loads the model and tokenizer if not already loaded."""
        if self.model is not None and model_name is None:
            return

        if model_name is not None:
            self.model_name = model_name

        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.working_dir,
            trust_remote_code=True
        )

        # Set pad token if not set -> TODO: remove?
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs : dict[str, Any] = {
            'cache_dir': self.working_dir,
            'trust_remote_code': True,
        }

        if self.load_in_8bit:
            model_kwargs['load_in_8bit'] = True
        elif self.load_in_4bit:
            model_kwargs['load_in_4bit'] = True
        elif self.device == 'cuda':
            model_kwargs['torch_dtype'] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        if not (self.load_in_8bit or self.load_in_4bit):
            self.model = self.model.to(self.device)

        self.model.eval()
        print("Model loaded successfully!")

    def load_model_from_checkpoint(self, checkpoint_path: Path):
        """
        Loads a fine-tuned checkpoint.

        :param checkpoint_path: Path to the checkpoint directory
        """
        print(f"Loading checkpoint from {checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

        if not (self.load_in_8bit or self.load_in_4bit):
            self.model = self.model.to(self.device)

        self.model.eval()

        print("Checkpoint loaded successfully!")

    def process(
            self,
            dataset: list[I],
            system_prompt: str,
            tools: list[dict],
            format_input: Callable[[I], str],
            parse_output: Callable[[str, str, float], O], # TODO: use the object returned by the model? If possible
            get_id: Callable[[I, int], str] = lambda x, idx: x.get('id', f'item-{idx}'),
    ) -> list[O]:
        """
        Processes a dataset through the model.

        :param dataset: List of items to process
        :param system_prompt: System prompt for all requests
        :param format_input: Function to convert each item to prompt
        :param parse_output: Function to parse model output. Receives (output_text, item_id)
        :param get_id: Function to generate IDs for each item
        :param tools: List of tool definitions (optional)
        :return: List of parsed outputs
        """
        if not dataset:
            return []

        self.load_model()

        results = []
        total = len(dataset)

        print(f"Processing {total} items sequentially")
        if tools:
            print(f"Tool calling enabled with {len(tools)} tool(s)")

        for idx, item in enumerate(dataset):
            item_id = get_id(item, idx)
            print(f"Processing item {idx+1}/{total} (id={item_id})")
            # Format prompt
            user_message = format_input(item)
            prompt = self._format_prompt(system_prompt, user_message, tools)

            # Generate response
            try:
                output, latency = self._generate(prompt)
                result = parse_output(output, item_id, latency)
            except Exception as e:
                print(f"Error processing item {item_id}: {e}")
                result = None

            results.append(result)

        successful = sum(1 for r in results if r is not None)
        print(f"\nCompleted: {successful}/{total} successful")

        return results

    def _format_prompt(
            self,
            system_prompt: str,
            user_message: str,
            tools: Optional[list[dict]] = None
    ) -> str:
        """Formats the prompt with system message, user message, and tools."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Add tool information to system prompt if provided
        if tools:
            tools_desc = "\n\nAvailable tools:\n"
            for tool in tools:
                tools_desc += f"- {tool['name']}: {tool['description']}\n"
                tools_desc += f"  Parameters: {json.dumps(tool['parameters'])}\n"
            messages[0]["content"] += tools_desc

        # If possible, use the chat_template tokenizer
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            return prompt

    def _generate(self, prompt: str) -> tuple[str, float]:
        """Generates response from the model."""
        start_time = time.time()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (excluding input)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip(), time.time() - start_time

    def fine_tune(
            self,
            train_dataset: list[dict],
            eval_dataset: list[dict],
            config: TrainingArguments,
            format_example: Optional[Callable[[dict], str]] = None,
    ) -> Path:
        """
        Fine-tunes the model on provided dataset.

        :param train_dataset: Training data (list of dicts with 'prompt' and 'completion' keys)
        :param eval_dataset: Evaluation data (optional)
        :param config: Fine-tuning configuration
        :param format_example: Function to format each example into text (optional)
        :return: Path to the fine-tuned model checkpoint
        """
        self.load_model()

        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFine-tuning configuration:")
        print(f"  Output dir: {config.output_dir}")
        print(f"  Epochs: {config.num_train_epochs}")
        print(f"  Batch size: {config.per_device_train_batch_size}")
        print(f"  Learning rate: {config.learning_rate}")

        # Prepare datasets
        train_data = self._prepare_dataset(train_dataset, format_example)
        eval_data = self._prepare_dataset(eval_dataset, format_example) if eval_dataset else None

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=config,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=data_collator,
        )

        # Train
        print("\nStarting fine-tuning...")
        trainer.train()

        # Save final model
        final_checkpoint = config.output_dir / "final"
        trainer.save_model(str(final_checkpoint))
        self.tokenizer.save_pretrained(str(final_checkpoint))

        print(f"\nFine-tuning complete! Model saved to {final_checkpoint}")

        return final_checkpoint

    def _prepare_dataset(
            self,
            dataset: list[dict],
            format_example: Optional[Callable[[dict], str]] = None
    ) -> Dataset:
        """Prepares and tokenizes dataset for training."""

        def default_format(example):
            """Default formatting: combines prompt and completion."""
            if 'prompt' in example and 'completion' in example:
                return f"{example['prompt']}\n{example['completion']}"
            elif 'text' in example:
                return example['text']
            else:
                raise ValueError("Example must have 'prompt'+'completion' or 'text' keys")

        formatter = format_example or default_format

        # Format texts
        texts = [formatter(example) for example in dataset]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=2048,
            padding=False,
        )

        # Add labels (same as input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'].copy()

        return Dataset.from_dict(tokenized)

