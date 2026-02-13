import os
import json

class LLMWrapper:
    """
    Unified LLM wrapper: supports local HuggingFace or remote API models.
    """

    def __init__(self, backend="local", model_name=None, api_key=None, device="cpu"):
        """
        backend: 'local' or 'openai'
        model_name: HF model path or API model name
        api_key: required if backend='openai'
        device: for local models ('cpu' or 'cuda')
        """
        self.backend = backend.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.device = device

        if self.backend == "local":
            self._load_local_model()
        elif self.backend == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI backend")
            import openai
            openai.api_key = api_key
            self.client = openai
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # -------------------------------
    # Local HuggingFace
    # -------------------------------
    def _load_local_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        if not self.model_name:
            raise ValueError("model_name required for local backend")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,      # HUGE VRAM saver
            device_map="auto",              # Let HF handle CUDA
            low_cpu_mem_usage=True
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    # -------------------------------
    # Unified generate
    # -------------------------------
    def generate(self, prompt, max_tokens=1024, temperature=0.1):
        if self.backend == "local":
            outputs = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                return_full_text=False,
            )
            return outputs[0]["generated_text"]
        elif self.backend == "openai":
            response = self.client.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text.strip()