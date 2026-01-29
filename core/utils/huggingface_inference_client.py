import time
from pathlib import Path
from typing import TypeVar, Generic, Callable, Optional, Any
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm

I = TypeVar('I')
O = TypeVar('O')


class HuggingFaceInferenceClient(Generic[I, O]):
    """
    Optimized client for high-performance inference with Hugging Face models.

    Key optimizations:
    - Flash Attention 2 for 2-4x speedup
    - Proper chat template handling
    - Quantization support (4-bit/8-bit)
    - torch.compile support
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
            use_flash_attention_2: bool = True,
            torch_compile: bool = False,
            hf_token: Optional[str] = None,
    ):
        """
        Creates an optimized inference client.

        :param model: Hugging Face model name or path
        :param device: Device to use ('cuda', 'cpu', or None for auto)
        :param max_output_tokens: Maximum tokens to generate
        :param temperature: Sampling temperature
        :param working_dir: Directory to cache models
        :param load_in_8bit: Load model in 8-bit precision
        :param load_in_4bit: Load model in 4-bit precision (recommended for RTX 2060)
        :param use_flash_attention_2: Use Flash Attention 2 (requires compatible GPU)
        :param torch_compile: Use torch.compile for extra speed
        :param hf_token: Hugging Face API token
        """
        self.model_name = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.working_dir = working_dir
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flash_attention_2 = use_flash_attention_2
        self.torch_compile = torch_compile
        self.hf_token = hf_token

        self.model = None
        self.tokenizer = None

        print(f"🚀 Initialized HuggingFaceInferenceClient for {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Flash Attention 2: {use_flash_attention_2}")
        print(f"   Quantization: {'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'None'}")

    def load_model(self, model_name: Optional[str] = None):
        """Loads the model and tokenizer with all optimizations."""
        if self.model is not None and model_name is None:
            return

        if model_name is not None:
            self.model_name = model_name

        print(f"\n📦 Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.working_dir,
            trust_remote_code=True,
            token=self.hf_token,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model loading configuration
        model_kwargs: dict[str, Any] = {
            'cache_dir': self.working_dir,
            'trust_remote_code': True,
            'token': self.hf_token,
            'low_cpu_mem_usage': True,
        }

        # Flash Attention 2 (huge speedup)
        if self.use_flash_attention_2 and self.device == 'cuda':
            model_kwargs['attn_implementation'] = "flash_attention_2"
            print("   ⚡ Enabling Flash Attention 2")

        # Quantization
        if self.load_in_4bit:
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("   📉 Using 4-bit quantization (NF4)")
        elif self.load_in_8bit:
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            print("   📉 Using 8-bit quantization")
        elif self.device == 'cuda':
            model_kwargs['torch_dtype'] = torch.bfloat16

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        if not (self.load_in_8bit or self.load_in_4bit):
            self.model = self.model.to(self.device)

        self.model.eval()

        # Torch compile (experimental, can give 20-30% speedup)
        if self.torch_compile and hasattr(torch, 'compile'):
            print("   🔧 Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        print("   ✓ Model loaded successfully!\n")

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
        Processes a dataset sequentially with proper chat template formatting.

        :param dataset: List of items to process
        :param system_prompt: System prompt
        :param tools: Tool definitions
        :param format_input: Function to convert each item to user message
        :param parse_output: Function to parse model output
        :param get_id: Function to generate IDs
        :return: List of parsed outputs
        """
        if not dataset:
            return []

        self.load_model()

        results = []
        total = len(dataset)

        print(f"🔄 Processing {total} items sequentially")

        # Process one by one
        for idx in tqdm(range(total), desc="Processing items"):
            item = dataset[idx]
            item_id = get_id(item, idx)
            user_message = format_input(item)

            try:
                output, latency = self._generate(
                    system_prompt=system_prompt,
                    tools=tools,
                    user_message=user_message
                )
                result = parse_output(output, item_id, latency)
                print(result)
            except Exception as e:
                print(f"\n❌ Error processing item {item_id}: {e}")
                result = None

            results.append(result)

        successful = sum(1 for r in results if r is not None)
        print(f"\n✅ Completed: {successful}/{total} successful")
        return results

    def _generate(
            self,
            system_prompt: str,
            tools: list[dict],
            user_message: str,
    ) -> tuple[str, float]:
        """
        Generates a response using proper chat template formatting.
        """
        # Build conversation
        total_start = time.time()
        t1 = time.time()
        system_content = system_prompt
        if tools:
            tools_desc = "\n\nAvailable tools:\n"
            for tool in tools:
                tools_desc += f"- {tool['name']}: {tool['description']}\n"
                tools_desc += f"  Parameters: {json.dumps(tool['parameters'])}\n"
            system_content += tools_desc

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        print(f"  [1] Message building: {time.time() - t1:.3f}s")

        start_time = time.time()

        # Use the model's chat template
        t2 = time.time()
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat templates
            prompt = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        print(f"  [2] Chat template: {time.time() - t2:.3f}s")
        print(f"  [2a] Prompt length: {len(prompt)} chars")

        # Tokenize
        t3 = time.time()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        print(f"  [3] Tokenization: {time.time() - t3:.3f}s")
        print(f"  [3a] Input tokens: {inputs.input_ids.shape[1]}")

        # Generate
        t4 = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        print(f"  [4] Generation: {time.time() - t4:.3f}s")
        print(f"  [4a] Output tokens: {outputs.shape[1]}")


        # Decode output
        t5 = time.time()
        prompt_len = inputs.attention_mask.sum().item()
        response = self.tokenizer.decode(
            outputs[0][prompt_len:],
            skip_special_tokens=False
        )

        print(f"  [5] Decoding: {time.time() - t5:.3f}s")
        print(f"  [5a] Response length: {len(response)} chars")

        latency = time.time() - total_start
        print(f"  [TOTAL]: {latency:.3f}s\n")


        latency = time.time() - start_time
        return response.strip(), latency
