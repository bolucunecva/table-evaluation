# llm/factory.py

def create_llm(backend: str, model_name: str, **kwargs):
    """
    Returns a model instance that implements `generate(prompt)`.
    backend: 'openai', 'hf', 'llama', etc.
    """
    if backend == "openai":
        from .api.openai_llm import OpenAILLM
        return OpenAILLM(model_name=model_name, **kwargs)
    elif backend == "hf":
        from .local.hf_llm import HFLocalLLM
        return HFLocalLLM(model_name=model_name, **kwargs)
    elif backend == "llama":
        from .local.llama_llm import LlamaLLM
        return LlamaLLM(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")