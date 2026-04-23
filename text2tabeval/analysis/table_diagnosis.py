"""
table_diagnosis.py
==================
Per-table diagnostic explanation of *why* a table scored low.

Two explanation modes are provided:

1. **Rule-based** (no LLM required) — uses ErrorTaxonomy detections and
   metric scores to generate a structured text report.

2. **LLM-based** (requires LLMWrapper) — uses your existing LLMWrapper to
   produce a richer natural-language explanation with suggested fixes.

Typical usage
-------------
>>> from text2tabeval.analysis.table_diagnosis import TableDiagnosis
>>> td = TableDiagnosis(pred_tables, gold_tables, scores_df)

# Rule-based (fast, no GPU needed)
>>> td.diagnose(table_idx=5)

# LLM-based (richer explanation)
>>> td.diagnose(table_idx=5, llm=llm_wrapper)

# Diagnose all lowest-scoring tables
>>> td.diagnose_worst(n=10, metric="TabXEval")
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from text2tabeval.analysis.error_taxonomy import ErrorTaxonomy, detect_errors


# ---------------------------------------------------------------------------
# Prompt template for LLM-based diagnosis
# ---------------------------------------------------------------------------

_DIAGNOSIS_PROMPT = """\
You are an expert evaluator of text-to-table generation systems.

You are given a PREDICTED table and a GOLD (reference) table, along with
automatic metric scores for this table pair.

Your task is to:
1. Identify what is wrong with the predicted table compared to the gold table.
2. Classify the error type(s): row_shuffle, missing_rows, extra_rows,
   cell_error, numeric_error, schema_mismatch, or perfect.
3. Explain in 2-3 sentences why the automatic metrics scored it the way they did.
4. Suggest one concrete fix the model should make.

Respond ONLY with a JSON object with these keys:
  "error_types": list of error type strings
  "explanation": string (2-3 sentences)
  "worst_metrics": list of metric names that are misleadingly high or low
  "suggested_fix": string (one sentence)

PREDICTED TABLE:
{pred}

GOLD TABLE:
{gold}

METRIC SCORES:
{scores}
"""


def _table_to_markdown(table: List[List[str]]) -> str:
    """Render a parsed table back as a Markdown string."""
    if not table:
        return "(empty)"
    rows = []
    for i, row in enumerate(table):
        rows.append("| " + " | ".join(str(c) for c in row) + " |")
        if i == 0:
            rows.append("| " + " | ".join("---" for _ in row) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TableDiagnosis:
    """
    Diagnose individual table pairs with rule-based or LLM explanations.

    Parameters
    ----------
    pred_tables, gold_tables : list of parsed tables
    scores_df : pd.DataFrame
        Shape ``(n_tables, n_metrics)``.
    """

    def __init__(
        self,
        pred_tables: List[List[List[str]]],
        gold_tables: List[List[List[str]]],
        scores_df: pd.DataFrame,
    ) -> None:
        if len(pred_tables) != len(gold_tables):
            raise ValueError("pred_tables and gold_tables must have the same length.")
        if len(pred_tables) != len(scores_df):
            raise ValueError("scores_df must have one row per table pair.")

        self.pred_tables = pred_tables
        self.gold_tables = gold_tables
        self.scores_df = scores_df

        # Normalise scores once
        self._normed = scores_df.copy()
        for col in self._normed.columns:
            mn, mx = self._normed[col].min(), self._normed[col].max()
            self._normed[col] = (self._normed[col] - mn) / (mx - mn + 1e-9)

        self._taxonomy = ErrorTaxonomy(pred_tables, gold_tables, scores_df)

    # ------------------------------------------------------------------
    # Rule-based diagnosis (no LLM)
    # ------------------------------------------------------------------

    def _rule_based_diagnosis(self, table_idx: int) -> Dict:
        """Internal rule-based diagnosis for one table."""
        pred = self.pred_tables[table_idx]
        gold = self.gold_tables[table_idx]
        errors = self._taxonomy._errors[table_idx]

        # Raw scores
        raw_scores = {
            col: round(float(self.scores_df.iloc[table_idx][col]), 4)
            for col in self.scores_df.columns
        }
        # Normalised scores
        norm_scores = {
            col: round(float(self._normed.iloc[table_idx][col]), 4)
            for col in self._normed.columns
        }

        # Identify misleadingly high metrics (high score but errors present)
        misleading_high = []
        misleading_low = []
        if "PERFECT" not in errors:
            for col, ns in norm_scores.items():
                if ns >= 0.7:
                    misleading_high.append(col)
                elif ns <= 0.2:
                    misleading_low.append(col)

        # Build explanation
        explanation_parts = []
        if "EMPTY_PRED" in errors:
            explanation_parts.append("The predicted table is empty.")
        if "SCHEMA_MISMATCH" in errors:
            pred_h = pred[0] if pred else []
            gold_h = gold[0] if gold else []
            explanation_parts.append(
                f"Column headers differ: predicted has {len(pred_h)} columns "
                f"({', '.join(str(c) for c in pred_h[:3])}...) vs gold "
                f"({', '.join(str(c) for c in gold_h[:3])}...)."
            )
        if "MISSING_ROWS" in errors:
            n_pred = len(pred) - 1
            n_gold = len(gold) - 1
            explanation_parts.append(
                f"Predicted table has {n_pred} data rows vs {n_gold} in gold "
                f"({n_gold - n_pred} rows missing)."
            )
        if "EXTRA_ROWS" in errors:
            n_pred = len(pred) - 1
            n_gold = len(gold) - 1
            explanation_parts.append(
                f"Predicted table has {n_pred - n_gold} extra rows compared to gold."
            )
        if "ROW_SHUFFLE" in errors:
            explanation_parts.append(
                "All rows are present but in a different order than the reference. "
                "Alignment-agnostic metrics (EM, ChrF, BERTScore) will score this "
                "lower than it deserves; alignment-aware metrics (H-Score, TabXEval) "
                "should handle it correctly."
            )
        if "CELL_ERROR" in errors:
            explanation_parts.append(
                "Some cell values do not match the reference. "
                "LLM-based metrics (TabEval, TabXEval) are most sensitive to this."
            )
        if "NUMERIC_ERROR" in errors:
            explanation_parts.append(
                "One or more numeric values differ from the reference beyond the "
                "5% tolerance. RMSE captures this; string-based metrics miss it."
            )
        if "PERFECT" in errors:
            explanation_parts.append(
                "No errors detected — predicted and gold tables appear equivalent."
            )

        if misleading_high:
            explanation_parts.append(
                f"Metrics {misleading_high} score this table high despite the errors "
                f"— these are likely alignment-agnostic or surface-form metrics."
            )

        # Suggested fix
        fix = "N/A"
        if "EMPTY_PRED" in errors:
            fix = "Check that the model output is parsed correctly before evaluation."
        elif "SCHEMA_MISMATCH" in errors:
            fix = "Align column headers between predicted and gold before evaluation, or verify the model generates the correct schema."
        elif "ROW_SHUFFLE" in errors:
            fix = "Sort both tables by a canonical key column before evaluation, or use alignment-aware metrics."
        elif "MISSING_ROWS" in errors:
            fix = "Investigate why the model generates fewer rows — it may truncate long inputs."
        elif "NUMERIC_ERROR" in errors:
            fix = "Add RMSE or numeric-specific metrics; string metrics will not catch numeric errors."
        elif "CELL_ERROR" in errors:
            fix = "Use TabXEval or TabEval for cell-level semantic correctness assessment."

        return {
            "table_idx": table_idx,
            "overall_score": round(
                float(self._normed.iloc[table_idx].mean(skipna=True)), 3
            ),
            "raw_scores": raw_scores,
            "error_types": sorted(errors),
            "explanation": " ".join(explanation_parts),
            "misleading_high_metrics": misleading_high,
            "misleading_low_metrics": misleading_low,
            "suggested_fix": fix,
            "pred_table_markdown": _table_to_markdown(pred),
            "gold_table_markdown": _table_to_markdown(gold),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(
        self,
        table_idx: int,
        llm=None,
    ) -> Dict:
        """
        Diagnose a single table pair.

        Parameters
        ----------
        table_idx : int
            Index into pred/gold tables.
        llm : LLMWrapper, optional
            If provided, an LLM call is made for a richer explanation.
            Falls back to rule-based if the LLM call fails.

        Returns
        -------
        dict with keys:
            ``table_idx``, ``overall_score``, ``raw_scores``, ``error_types``,
            ``explanation``, ``misleading_high_metrics``,
            ``misleading_low_metrics``, ``suggested_fix``,
            ``pred_table_markdown``, ``gold_table_markdown``,
            and (if LLM used) ``llm_explanation``.
        """
        result = self._rule_based_diagnosis(table_idx)

        if llm is not None:
            result["llm_explanation"] = self._llm_diagnosis(table_idx, llm)

        return result

    def _llm_diagnosis(self, table_idx: int, llm) -> Dict:
        """Call LLMWrapper for a richer explanation."""
        pred = self.pred_tables[table_idx]
        gold = self.gold_tables[table_idx]
        scores = {
            col: round(float(self.scores_df.iloc[table_idx][col]), 4)
            for col in self.scores_df.columns
        }

        prompt = _DIAGNOSIS_PROMPT.format(
            pred=_table_to_markdown(pred),
            gold=_table_to_markdown(gold),
            scores=json.dumps(scores, indent=2),
        )

        try:
            raw = llm.generate(prompt)
            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as exc:
            return {"error": f"LLM diagnosis failed: {exc}"}

    # ------------------------------------------------------------------
    # Batch diagnosis of worst-scoring tables
    # ------------------------------------------------------------------

    def diagnose_worst(
        self,
        n: int = 10,
        metric: Optional[str] = None,
        llm=None,
    ) -> List[Dict]:
        """
        Diagnose the ``n`` lowest-scoring table pairs.

        Parameters
        ----------
        n : int
            Number of tables to diagnose (default 10).
        metric : str, optional
            Column name in scores_df to sort by. If None, uses the mean
            across all metrics.
        llm : LLMWrapper, optional
            If provided, uses LLM-based explanation.

        Returns
        -------
        list of diagnosis dicts, sorted worst-first.
        """
        if metric is not None and metric not in self.scores_df.columns:
            raise ValueError(
                f"Metric '{metric}' not in scores_df. "
                f"Available: {list(self.scores_df.columns)}"
            )

        if metric:
            sort_series = self._normed[metric]
        else:
            sort_series = self._normed.mean(axis=1)

        worst_indices = sort_series.nsmallest(n).index.tolist()
        return [self.diagnose(i, llm=llm) for i in worst_indices]

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_diagnosis(self, table_idx: int, llm=None) -> None:
        """Print a human-readable diagnosis for one table."""
        d = self.diagnose(table_idx, llm=llm)
        print("=" * 60)
        print(f"TABLE DIAGNOSIS  —  Index: {d['table_idx']}")
        print("=" * 60)
        print(f"Overall normalised score : {d['overall_score']:.3f}")
        print(f"Error types detected     : {', '.join(d['error_types'])}")
        print(f"\nExplanation:\n  {d['explanation']}")
        print(f"\nSuggested fix:\n  {d['suggested_fix']}")
        if d.get("misleading_high_metrics"):
            print(f"\nMisleadingly high metrics: {d['misleading_high_metrics']}")
        print("\nRaw metric scores:")
        for k, v in d["raw_scores"].items():
            print(f"  {k:30s}: {v:.4f}")
        if "llm_explanation" in d:
            print("\nLLM Explanation:")
            print(json.dumps(d["llm_explanation"], indent=2))
        print("\nPredicted Table:")
        print(d["pred_table_markdown"])
        print("\nGold Table:")
        print(d["gold_table_markdown"])
