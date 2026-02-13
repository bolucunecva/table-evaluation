from pathlib import Path
from text2tabeval.llm.factory import create_llm
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def create_nli_pipeline(nli_model_name="roberta-large-mnli"):
    tokenizer = AutoTokenizer.from_pretrained(nli_model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        nli_model_name,
        use_safetensors=False   # ← correct flag
    )

    nli_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=-1   # force CPU
    )

    return nli_pipeline


# -------------------------
# Stage 1: Unroll Markdown table to atomic statements
# -------------------------
def unroll_table(llm, table_markdown, prompt_path=None):
    if prompt_path is None:
        prompt_path = Path(__file__).parent.parent / "prompts" / "tabeval.txt"
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    prompt = prompt_template.format(table_markdown=table_markdown)
    output = llm.generate(prompt)
    statements = [line.strip() for line in output.split("\n") if line.strip()]
    return statements


# -------------------------
# Stage 2: Entailment scoring
# -------------------------
def entailment_score(statement1, statement2, nli_pipeline):
    """
    Compute entailment probability of statement1 -> statement2
    """
    result = nli_pipeline(
        [(statement1, statement2)],
        truncation=True,
        padding=True,
        max_length=512
    )
    for r in result[0]:
        if r['label'].upper() == "ENTAILMENT":
            return r['score']
    return 0.0


def compute_prf(pred_statements, gold_statements, nli_pipeline):
    """Compute precision, recall, F1 for a pair of statement sets"""
    # Precision: pred -> gold
    precision_scores = [max([entailment_score(pi, gj, nli_pipeline) for gj in gold_statements] or [0.0])
                        for pi in pred_statements]
    precision = np.mean(precision_scores) if precision_scores else 0.0

    # Recall: gold -> pred
    recall_scores = [max([entailment_score(pi, gj, nli_pipeline) for pi in pred_statements] or [0.0])
                     for gj in gold_statements]
    recall = np.mean(recall_scores) if recall_scores else 0.0

    # F1
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# -------------------------
# Full evaluation
# -------------------------
def evaluate_table_pair(pred_table_md, gold_table_md, llm, nli_pipeline, prompt_path=None):
    pred_statements = unroll_table(llm, pred_table_md, prompt_path)
    gold_statements = unroll_table(llm, gold_table_md, prompt_path)
    precision, recall, f1 = compute_prf(pred_statements, gold_statements, nli_pipeline)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_statements": pred_statements,
        "gold_statements": gold_statements
    }


def evaluate_tables(pred_tables, gold_tables, llm, nli_model="roberta-large-mnli", prompt_path=None, device=None):
    """Evaluate multiple table pairs and aggregate results"""
    nli_pipeline = create_nli_pipeline(nli_model)

    precisions, recalls, f1s = [], [], []
    all_results = []

    for pred_md, gold_md in zip(pred_tables, gold_tables):
        res = evaluate_table_pair(pred_md, gold_md, llm, nli_pipeline, prompt_path)
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        f1s.append(res["f1"])
        all_results.append(res)

    aggregated = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0
    }

    return aggregated