# text2tabeval/__init__.py
from text2tabeval.core.evaluator import Evaluator

def evaluate(pred_table, gold_table, methods=None):
    evaluator = Evaluator(methods)
    return evaluator.evaluate(pred_table, gold_table)

__all__ = ["evaluate", "Evaluator"]