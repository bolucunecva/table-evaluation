# core/evaluator.py
from text2tabeval.core.registry import METRICS
from text2tabeval.llm.factory import create_llm

class Evaluator:
    def __init__(self, methods, llm_config=None):
        self.methods = methods
        self.llm = create_llm(**llm_config) if llm_config else None

    def evaluate(self, pred, gold):
        results = {}

        for name in self.methods:
            metric_cls = METRICS[name]

            if metric_cls.is_llm_based:
                metric = metric_cls(llm=self.llm)
            else:
                metric = metric_cls()

            results.update(metric.evaluate(pred, gold))

        return results