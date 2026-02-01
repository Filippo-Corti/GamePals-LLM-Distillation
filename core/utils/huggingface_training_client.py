import json
from pathlib import Path
from typing import TypeVar, Generic, Callable, Optional, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

T = TypeVar('T')


class HuggingFaceTrainingClient(Generic[T]):
    """
    Simple client for full-precision fine-tuning.
    """

    def __init__(
            self,
            model: str,
            device: Optional[str] = None,
            working_dir: Path = Path("./data/huggingface"),
            use_flash_attention_2: bool = True,
            hf_token: Optional[str] = None,
    ):
        """
        Creates a training client.

        :param model: Hugging Face model name or path
        :param device: Device to use ('cuda', 'cpu', or None for auto)
        :param working_dir: Directory to cache models
        :param use_flash_attention_2: Use Flash Attention 2 (A100 optimization)
        :param hf_token: Hugging Face API token
        """
        self.model_name = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.working_dir = working_dir
        self.use_flash_attention_2 = use_flash_attention_2
        self.hf_token = hf_token

        self.model = None
        self.tokenizer = None

        print(f"🏋️  Initialized HuggingFaceTrainingClient for {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Flash Attention 2: {use_flash_attention_2}")

    def load_tokenizer(self):
        """Loads the tokenizer."""
        if self.tokenizer is not None:
            return

        print(f"\n📦 Loading tokenizer: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.working_dir,
            trust_remote_code=True,
            token=self.hf_token,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("   ✓ Tokenizer loaded successfully!\n")

    def load_model_for_training(self):
        """Loads the model for training."""
        self.load_tokenizer()

        if self.model is not None:
            return

        print(f"\n📦 Loading model for training: {self.model_name}")

        # Model configuration
        model_kwargs: dict[str, Any] = {
            'cache_dir': self.working_dir,
            'trust_remote_code': True,
            'token': self.hf_token,
            'low_cpu_mem_usage': True,
            'torch_dtype': torch.bfloat16,  # Mixed precision for efficiency
        }

        # Flash Attention 2 (A100 optimization)
        if self.use_flash_attention_2 and self.device == 'cuda':
            model_kwargs['attn_implementation'] = "flash_attention_2"
            print("   ⚡ Enabling Flash Attention 2")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

        print("   ✓ Model loaded successfully!\n")

    def fine_tune(
            self,
            train_dataset: list[T],
            eval_dataset: Optional[list[T]],
            format_example: Callable[[T, AutoTokenizer], str],
            output_dir: Path = Path("./checkpoints"),
            num_epochs: int = 3,
            batch_size: int = 8,
            gradient_accumulation_steps: int = 2,
            learning_rate: float = 2e-4,
            max_seq_length: int = 2048,
            save_steps: int = 100,
            eval_steps: int = 100,
            warmup_ratio: float = 0.03,
    ) -> Path:
        """
        Fine-tunes the model.

        :param train_dataset: Training data
        :param eval_dataset: Evaluation data
        :param format_example: Function to format examples
        :param output_dir: Directory to save checkpoints
        :param num_epochs: Number of training epochs
        :param batch_size: Per-device batch size
        :param gradient_accumulation_steps: Gradient accumulation steps
        :param learning_rate: Learning rate
        :param max_seq_length: Maximum sequence length
        :param save_steps: Save checkpoint every N steps
        :param eval_steps: Evaluate every N steps
        :param warmup_ratio: Warmup ratio for learning rate scheduler
        :return: Path to final checkpoint
        """
        # Load tokenizer first
        self.load_tokenizer()

        # Load the model
        self.load_model_for_training()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔧 Fine-tuning configuration:")
        print(f"   Output dir: {output_dir}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Gradient accumulation: {gradient_accumulation_steps}")
        print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Max sequence length: {max_seq_length}")

        # Format datasets as text
        print(f"📝 Formatting {len(train_dataset)} training examples...")
        train_texts = [format_example(example, self.tokenizer) for example in train_dataset]

        if eval_dataset:
            print(f"📝 Formatting {len(eval_dataset)} eval examples...")
            eval_texts = [format_example(example, self.tokenizer) for example in eval_dataset]
        else:
            eval_texts = None

        # Create datasets
        print(f"🔤 Creating and tokenizing datasets...")
        train_data = Dataset.from_dict({"text": train_texts})

        def tokenize_function(examples):
            """Tokenize examples."""
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length,
            )

        train_data = train_data.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing training data"
        )

        if eval_texts:
            eval_data = Dataset.from_dict({"text": eval_texts})
            eval_data = eval_data.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                desc="Tokenizing eval data"
            )
        else:
            eval_data = None

        print(f"   ✓ Datasets prepared")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,

            # Optimizer settings
            optim="adamw_torch_fused",  # Fast optimizer for A100
            learning_rate=learning_rate,
            weight_decay=0.01,
            max_grad_norm=1.0,

            # Learning rate schedule
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,

            # Mixed precision (bfloat16 is standard)
            bf16=True,
            fp16=False,

            # Logging
            logging_steps=10,
            logging_strategy="steps",

            # Evaluation
            eval_strategy="steps" if eval_data else "no",
            eval_steps=eval_steps if eval_data else None,

            # Saving
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,

            # Performance
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            # Misc
            report_to="none",
            seed=42,
            data_seed=42,
        )

        # Data collator (automatically creates labels from input_ids)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
            pad_to_multiple_of=8,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=data_collator,
        )

        # Train
        print("\n🚀 Starting fine-tuning...\n")
        trainer.train()

        # Save final model
        final_checkpoint = output_dir / "final"
        trainer.save_model(str(final_checkpoint))
        self.tokenizer.save_pretrained(str(final_checkpoint))

        print(f"\n✅ Fine-tuning complete! Model saved to {final_checkpoint}")

        return final_checkpoint