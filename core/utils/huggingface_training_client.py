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
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset

T = TypeVar('T')


class HuggingFaceTrainingClient(Generic[T]):
    """
    Optimized client for fine-tuning with LoRA/QLoRA.

    Key optimizations:
    - QLoRA for efficient training on RTX 2060 and A100
    - Flash Attention 2 for 2-4x training speedup
    - Gradient checkpointing for memory efficiency
    - Paged optimizers for large batches
    - Automatic mixed precision (bfloat16)
    """

    def __init__(
            self,
            model: str,
            device: Optional[str] = None,
            working_dir: Path = Path("./data/huggingface"),
            use_qlora: bool = True,
            use_flash_attention_2: bool = True,
            hf_token: Optional[str] = None,
    ):
        """
        Creates an optimized training client.

        :param model: Hugging Face model name or path
        :param device: Device to use ('cuda', 'cpu', or None for auto)
        :param working_dir: Directory to cache models
        :param use_qlora: Use QLoRA (4-bit quantization + LoRA) for efficient training
        :param use_flash_attention_2: Use Flash Attention 2
        :param hf_token: Hugging Face API token
        """
        self.model_name = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.working_dir = working_dir
        self.use_qlora = use_qlora
        self.use_flash_attention_2 = use_flash_attention_2
        self.hf_token = hf_token

        self.model = None
        self.tokenizer = None

        print(f"🏋️  Initialized HuggingFaceTrainingClient for {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   QLoRA: {use_qlora}")
        print(f"   Flash Attention 2: {use_flash_attention_2}")

    def load_tokenizer(self):
        """Loads just the tokenizer (lightweight, no model)."""
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
        """Loads the model optimized for training."""
        # Load tokenizer first if not already loaded
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
        }

        # Flash Attention 2
        if self.use_flash_attention_2 and self.device == 'cuda':
            model_kwargs['attn_implementation'] = "flash_attention_2"
            print("   ⚡ Enabling Flash Attention 2")

        # QLoRA configuration (recommended)
        if self.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.bfloat16,  # Store in bfloat16
            )
            model_kwargs['quantization_config'] = bnb_config
            print("   📉 Using QLoRA (4-bit quantization)")
        else:
            # Full precision training
            model_kwargs['torch_dtype'] = torch.bfloat16

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        # Prepare for k-bit training (required for QLoRA)
        if self.use_qlora:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True
            )
            print("   ✓ Model prepared for QLoRA training")
        else:
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
            batch_size: int = 4,
            gradient_accumulation_steps: int = 4,
            learning_rate: float = 2e-4,
            max_seq_length: int = 2048,
            lora_r: int = 64,
            lora_alpha: int = 16,
            lora_dropout: float = 0.05,
            save_steps: int = 100,
            eval_steps: int = 100,
            warmup_ratio: float = 0.03,
    ) -> Path:
        """
        Fine-tunes the model using LoRA/QLoRA.

        :param train_dataset: Training data
        :param eval_dataset: Evaluation data
        :param format_example: Function to format examples (has access to self.tokenizer)
        :param output_dir: Directory to save checkpoints
        :param num_epochs: Number of training epochs
        :param batch_size: Per-device batch size
        :param gradient_accumulation_steps: Gradient accumulation steps
        :param learning_rate: Peak learning rate
        :param max_seq_length: Maximum sequence length
        :param lora_r: LoRA rank (higher = more parameters, better quality, slower)
        :param lora_alpha: LoRA alpha (scaling factor)
        :param lora_dropout: LoRA dropout
        :param save_steps: Save checkpoint every N steps
        :param eval_steps: Evaluate every N steps
        :param warmup_ratio: Warmup ratio for learning rate scheduler
        :return: Path to final checkpoint
        """
        # Load tokenizer first so format_example can use it
        self.load_tokenizer()

        # Now load the full model
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

        if self.use_qlora:
            print(f"   LoRA rank: {lora_r}")
            print(f"   LoRA alpha: {lora_alpha}")
            print(f"   LoRA dropout: {lora_dropout}")

        # Apply LoRA
        if self.use_qlora:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],  # Targets all attention and FFN layers
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        # Format datasets as text
        print(f"📝 Formatting {len(train_dataset)} training examples...")
        train_texts = [format_example(example, self.tokenizer) for example in train_dataset]

        if eval_dataset:
            print(f"📝 Formatting {len(eval_dataset)} eval examples...")
            eval_texts = [format_example(example, self.tokenizer) for example in eval_dataset]
        else:
            eval_texts = None

        # Create datasets from text first
        print(f"🔤 Creating and tokenizing datasets...")
        train_data = Dataset.from_dict({"text": train_texts})

        def tokenize_function(examples):
            """Tokenize examples - labels will be created by DataCollator."""
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
            optim="paged_adamw_8bit" if self.use_qlora else "adamw_torch_fused",
            learning_rate=learning_rate,
            weight_decay=0.01,
            max_grad_norm=1.0,

            # Learning rate schedule
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,

            # Mixed precision
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

            # Performance optimizations
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            # Misc
            report_to="none",
            seed=42,
            data_seed=42,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        # Use standard Trainer
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

    def load_checkpoint(self, checkpoint_path: Path):
        """Loads a LoRA checkpoint for further training or inference."""
        from peft import PeftModel

        print(f"📦 Loading checkpoint from {checkpoint_path}")

        self.load_model_for_training()

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(
            self.model,
            str(checkpoint_path),
            is_trainable=True,
        )

        print("   ✓ Checkpoint loaded successfully!")

    def merge_and_save(self, checkpoint_path: Path, output_path: Path):
        """Merges LoRA weights with base model and saves."""
        from peft import PeftModel

        print(f"🔀 Merging LoRA weights from {checkpoint_path}")

        # Load base model
        self.load_model_for_training()

        # Load LoRA weights
        model = PeftModel.from_pretrained(
            self.model,
            str(checkpoint_path),
        )

        # Merge
        merged_model = model.merge_and_unload()

        # Save
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        merged_model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

        print(f"   ✓ Merged model saved to {output_path}")