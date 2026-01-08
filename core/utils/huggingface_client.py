import json
import time
from pathlib import Path
from typing import TypeVar, Generic, Callable, Optional, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import Dataset
from tqdm import tqdm

I = TypeVar('I')
O = TypeVar('O')


class HuggingFaceClient(Generic[I, O]):
    """
    Optimized client for processing datasets and fine-tuning models with Hugging Face.

    Supports local inference, LoRA fine-tuning, batching, and multiple optimizations.
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
            use_better_transformer: bool = True,
            torch_compile: bool = False,
            hf_token: Optional[str] = None,
    ):
        """
        Creates an optimized HuggingFaceClient instance.

        :param model: Hugging Face model name or path
        :param device: Device to use ('cuda', 'cpu', or None for auto)
        :param max_output_tokens: Maximum tokens to generate (default: 512)
        :param temperature: Sampling temperature (default: 1.0)
        :param working_dir: Directory to cache models
        :param load_in_8bit: Load model in 8-bit precision (default: False)
        :param load_in_4bit: Load model in 4-bit precision (default: False)
        :param use_better_transformer: Use BetterTransformer optimization (default: True)
        :param torch_compile: Use torch.compile for extra speed (default: False)
        :param hf_token: Hugging Face API token for gated models
        """
        self.model_name = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.working_dir = working_dir
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_better_transformer = use_better_transformer
        self.torch_compile = torch_compile
        self.hf_token = hf_token

        self.model = None
        self.tokenizer = None
        self.is_compiled = False

        # System prompt caching
        self.cached_system_prompt = None
        self.cached_system_prefix = None

        print(f"Initialized HuggingFaceClient for {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Optimizations: BetterTransformer={use_better_transformer}, Compile={torch_compile}")

    def load_model(self, model_name: Optional[str] = None):
        """Loads the model and tokenizer with all optimizations."""
        if self.model is not None and model_name is None:
            return

        if model_name is not None:
            self.model_name = model_name

        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.working_dir,
            trust_remote_code=True,
            token=self.hf_token,
        )

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Configure model loading
        model_kwargs: dict[str, Any] = {
            'cache_dir': self.working_dir,
            'trust_remote_code': True,
            'token': self.hf_token,
            'low_cpu_mem_usage': True,
        }

        # Quantization config
        if self.load_in_4bit:
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.load_in_8bit:
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif self.device == 'cuda':
            model_kwargs['dtype'] = torch.float16

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # Move to device if not quantized
        if not (self.load_in_8bit or self.load_in_4bit):
            self.model = self.model.to(self.device)

        # Apply BetterTransformer (only works with certain models)
        if self.use_better_transformer and not (self.load_in_8bit or self.load_in_4bit):
            try:
                self.model = self.model.to_bettertransformer()
                print("BetterTransformer enabled")
            except Exception as e:
                print(f"BetterTransformer not available: {e}")

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        self.model.eval()

        print("✓ Model loaded successfully!")

    def load_model_from_checkpoint(self, checkpoint_path: Path):
        """Loads a fine-tuned checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            token=self.hf_token
        )

        model_kwargs = {
            'token': self.hf_token,
            'low_cpu_mem_usage': True,
        }

        if self.device == 'cuda':
            model_kwargs['dtype'] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            **model_kwargs
        )

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
            parse_output: Callable[[str, str, float], O],
            get_id: Callable[[I, int], str] = lambda x, idx: x.get('id', f'item-{idx}'),
    ) -> list[O]:
        """
        Processes a dataset through the model sequentially with system prompt caching.

        :param dataset: List of items to process
        :param system_prompt: System prompt for all requests (will be cached)
        :param tools: List of tool definitions
        :param format_input: Function to convert each item to prompt
        :param parse_output: Function to parse model output (output_text, item_id, latency)
        :param get_id: Function to generate IDs for each item
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

        # Cache the system prompt prefix once
        self._cache_system_prompt(system_prompt, tools)
        print(f"System prompt cached for reuse across {total} items")

        # Process one by one
        for idx in tqdm(range(total), desc="Processing items"):
            item = dataset[idx]
            item_id = get_id(item, idx)

            user_message = format_input(item)

            # Generate response using cached system prompt
            try:
                output, latency = self._generate_with_cache(user_message)
                result = parse_output(output, item_id, latency)
            except Exception as e:
                print(f"Error processing item {item_id}: {e}")
                result = None

            results.append(result)

        successful = sum(1 for r in results if r is not None)
        print(f"\n✓ Completed: {successful}/{total} successful")

        return results

    def _cache_system_prompt(
            self,
            system_prompt: str,
            tools: Optional[list[dict]] = None
    ):
        """
        Caches the system prompt prefix to avoid re-encoding for every request.
        This stores the formatted system prompt string for reuse.
        """
        # Create a cache key from system prompt and tools
        cache_key = (system_prompt, json.dumps(tools) if tools else None)

        if self.cached_system_prompt == cache_key:
            return  # Already cached

        # Build the system message
        system_content = system_prompt
        if tools:
            tools_desc = "\n\nAvailable tools:\n"
            for tool in tools:
                tools_desc += f"- {tool['name']}: {tool['description']}\n"
                tools_desc += f"  Parameters: {json.dumps(tool['parameters'])}\n"
            system_content += tools_desc

        # Format the system part of the prompt
        messages = [{"role": "system", "content": system_content}]

        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use chat template to format system message
            self.cached_system_prefix = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback formatting
            self.cached_system_prefix = f"System: {system_content}\n\n"

        self.cached_system_prompt = cache_key

    def _generate_with_cache(self, user_message: str) -> tuple[str, float]:
        """
        Generates response using the cached system prompt prefix.
        Only formats and encodes the user message + combines with cached prefix.
        """
        start_time = time.time()

        # Build the full prompt using cached system prefix
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use chat template for the user message part
            messages = [
                {"role": "user", "content": user_message}
            ]
            user_part = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Combine cached system with user part
            # Note: This is a simple concatenation; for better caching we'd need
            # to cache the actual token embeddings, but that's model-specific
            full_prompt = self.cached_system_prefix + user_part
        else:
            # Fallback: simple string concatenation
            full_prompt = self.cached_system_prefix + f"User: {user_message}\n\nAssistant:"

        # Tokenize the full prompt
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
            )

        sequences = outputs.sequences
        prompt_len = inputs["attention_mask"].sum(dim=1).item()
        response = self.tokenizer.decode(
            sequences[0][prompt_len:],
            skip_special_tokens=True
        )

        return response.strip(), time.time() - start_time

    def _format_prompt(
            self,
            system_prompt: str,
            user_message: str,
            tools: Optional[list[dict]] = None
    ) -> str:
        """
        Formats the prompt with system message, user message, and tools.
        Note: This is kept for backward compatibility but not used in the main process loop.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Add tool information
        if tools:
            tools_desc = "\n\nAvailable tools:\n"
            for tool in tools:
                tools_desc += f"- {tool['name']}: {tool['description']}\n"
                tools_desc += f"  Parameters: {json.dumps(tool['parameters'])}\n"
            messages[0]["content"] += tools_desc

        # Use chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            return f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"

    def fine_tune(
            self,
            train_dataset: list[dict],
            eval_dataset: list[dict],
            config: TrainingArguments,
            format_example: Optional[Callable[[dict], str]] = None,
    ) -> Path:
        """
        Fine-tunes the model (full fine-tuning).

        :param train_dataset: Training data
        :param eval_dataset: Evaluation data
        :param config: Training configuration
        :param format_example: Function to format examples
        :return: Path to fine-tuned model
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

        # Data collator with dynamic padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # Optimize for GPU
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

        # Save model
        final_checkpoint = out_dir / "final"
        trainer.save_model(str(final_checkpoint))
        self.tokenizer.save_pretrained(str(final_checkpoint))

        print(f"\n✓ Fine-tuning complete! Model saved to {final_checkpoint}")

        return final_checkpoint

    def _prepare_dataset(
            self,
            dataset: list[dict],
            format_example: Optional[Callable[[dict], str]] = None
    ) -> Dataset:
        """Prepares and tokenizes dataset efficiently."""

        def default_format(example):
            if 'prompt' in example and 'completion' in example:
                return f"{example['prompt']}\n{example['completion']}"
            elif 'text' in example:
                return example['text']
            else:
                raise ValueError("Example must have 'prompt'+'completion' or 'text' keys")

        formatter = format_example or default_format

        # Format texts
        texts = [formatter(example) for example in dataset]

        # Tokenize in batches for speed
        print(f"Tokenizing {len(texts)} examples...")
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=4096,
            padding=False,
            return_attention_mask=False,  # Save memory
        )

        # Add labels
        tokenized['labels'] = [ids.copy() for ids in tokenized['input_ids']]

        return Dataset.from_dict(tokenized)

    def unload_model(self):
        """Unloads the model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print("✓ Model unloaded from memory")