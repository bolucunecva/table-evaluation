# core/base.py
from abc import ABC, abstractmethod

class TableMetric(ABC):
    name: str
    is_llm_based: bool = False

    @abstractmethod
    def evaluate(self, pred, gold) -> dict:
        pass
