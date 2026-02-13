# metrics/llm/base.py
from text2tabeval.core.base import TableMetric

class LLMMetric(TableMetric):
    is_llm_based = True

    def __init__(self, llm=None, **kwargs):
        if llm is None:
            raise ValueError("LLMMetric requires an LLM instance")
        self.llm = llm