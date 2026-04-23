"""
error_taxonomy.py
=================
Rule-based classification of T2T generation errors.

For each (predicted, gold) table pair the module detects which error
types are present, then builds a *metric sensitivity table* that shows
how well each metric catches each error type.

Error types
-----------
ROW_SHUFFLE     Correct cells but rows in wrong order
MISSING_ROWS    Fewer rows than reference
EXTRA_ROWS      More rows than reference
CELL_ERROR      Individual cell values differ
NUMERIC_ERROR   Numeric values differ beyond tolerance
SCHEMA_MISMATCH Column headers differ
EMPTY_PRED      Predicted table is empty
PERFECT         No errors detected

Typical usage
-------------
>>> from text2tabeval.analysis.error_taxonomy import ErrorTaxonomy
>>> et = ErrorTaxonomy(pred_tables, gold_tables, scores_df)
>>> et.get_error_distribution()           # Counter of error types
>>> et.metric_sensitivity_report()        # DataFrame: metrics × error types
>>> et.annotated_table(table_idx=5)       # dict with errors for one table
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Error type constants
# ---------------------------------------------------------------------------

class ErrorType:
    PERFECT         = "PERFECT"
    EMPTY_PRED      = "EMPTY_PRED"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    MISSING_ROWS    = "MISSING_ROWS"
    EXTRA_ROWS      = "EXTRA_ROWS"
    ROW_SHUFFLE     = "ROW_SHUFFLE"
    CELL_ERROR      = "CELL_ERROR"
    NUMERIC_ERROR   = "NUMERIC_ERROR"

ALL_ERROR_TYPES = [
    ErrorType.PERFECT,
    ErrorType.EMPTY_PRED,
    ErrorType.SCHEMA_MISMATCH,
    ErrorType.MISSING_ROWS,
    ErrorType.EXTRA_ROWS,
    ErrorType.ROW_SHUFFLE,
    ErrorType.CELL_ERROR,
    ErrorType.NUMERIC_ERROR,
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_numeric(val: str) -> bool:
    try:
        float(val.replace(",", "").replace("%", ""))
        return True
    except ValueError:
        return False


def _normalise_cell(val: str) -> str:
    return val.strip().lower()


def _table_to_cell_set(table: List[List[str]]) -> Set[str]:
    """Flat set of normalised cell values (ignoring row position)."""
    cells = set()
    for row in table:
        for cell in row:
            cells.add(_normalise_cell(cell))
    return cells


def _table_headers(table: List[List[str]]) -> List[str]:
    """Return first row (assumed header)."""
    if not table:
        return []
    return [_normalise_cell(c) for c in table[0]]


def _data_rows(table: List[List[str]]) -> List[List[str]]:
    """Return all rows except the header."""
    return table[1:] if len(table) > 1 else []


# ---------------------------------------------------------------------------
# Per-table error detection
# ---------------------------------------------------------------------------

def detect_errors(
    pred: List[List[str]],
    gold: List[List[str]],
    numeric_tol: float = 0.05,
) -> Set[str]:
    """
    Detect error types in a single (pred, gold) table pair.

    Parameters
    ----------
    pred, gold : list of list of str
        Parsed tables (row-major, first row = header).
    numeric_tol : float
        Relative tolerance for numeric comparison (default 5 %).

    Returns
    -------
    set of ErrorType constants
    """
    errors: Set[str] = set()

    # --- Empty prediction ---
    if not pred or all(not row for row in pred):
        return {ErrorType.EMPTY_PRED}

    # --- Schema / header mismatch ---
    pred_headers = _table_headers(pred)
    gold_headers = _table_headers(gold)
    if pred_headers != gold_headers:
        errors.add(ErrorType.SCHEMA_MISMATCH)

    # --- Row count differences ---
    pred_data = _data_rows(pred)
    gold_data = _data_rows(gold)

    if len(pred_data) < len(gold_data):
        errors.add(ErrorType.MISSING_ROWS)
    elif len(pred_data) > len(gold_data):
        errors.add(ErrorType.EXTRA_ROWS)

    # --- Row shuffle: same cells present but different row order ---
    pred_row_strs = ["|".join(_normalise_cell(c) for c in r) for r in pred_data]
    gold_row_strs = ["|".join(_normalise_cell(c) for c in r) for r in gold_data]

    pred_row_set = set(pred_row_strs)
    gold_row_set = set(gold_row_strs)

    if (
        pred_row_set == gold_row_set          # same rows
        and pred_row_strs != gold_row_strs    # different order
        and ErrorType.MISSING_ROWS not in errors
        and ErrorType.EXTRA_ROWS not in errors
    ):
        errors.add(ErrorType.ROW_SHUFFLE)

    # --- Cell-level errors ---
    pred_cells = _table_to_cell_set(pred)
    gold_cells = _table_to_cell_set(gold)
    missing_cells = gold_cells - pred_cells
    if missing_cells - {""} :
        errors.add(ErrorType.CELL_ERROR)

    # --- Numeric errors ---
    for gold_row in gold_data:
        for cell in gold_row:
            if _is_numeric(cell):
                gold_val = float(cell.replace(",", "").replace("%", ""))
                # Check if a close-enough numeric value exists in pred
                found_close = False
                for pred_row in pred_data:
                    for pcell in pred_row:
                        if _is_numeric(pcell):
                            pred_val = float(pcell.replace(",", "").replace("%", ""))
                            if gold_val == 0:
                                if abs(pred_val) < 1e-9:
                                    found_close = True
                            elif abs(pred_val - gold_val) / abs(gold_val) <= numeric_tol:
                                found_close = True
                if not found_close:
                    errors.add(ErrorType.NUMERIC_ERROR)
                    break
        if ErrorType.NUMERIC_ERROR in errors:
            break

    if not errors:
        errors.add(ErrorType.PERFECT)

    return errors


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ErrorTaxonomy:
    """
    Error taxonomy analysis for a full dataset.

    Parameters
    ----------
    pred_tables : list of parsed tables
    gold_tables : list of parsed tables
    scores_df : pd.DataFrame, optional
        Shape ``(n_tables, n_metrics)``. Required for metric sensitivity.
    """

    def __init__(
        self,
        pred_tables: List[List[List[str]]],
        gold_tables: List[List[List[str]]],
        scores_df: Optional[pd.DataFrame] = None,
    ) -> None:
        if len(pred_tables) != len(gold_tables):
            raise ValueError(
                f"Length mismatch: {len(pred_tables)} pred vs {len(gold_tables)} gold."
            )
        self.pred_tables = pred_tables
        self.gold_tables = gold_tables
        self.scores_df = scores_df

        # Compute errors for all pairs upfront
        self._errors: List[Set[str]] = [
            detect_errors(p, g)
            for p, g in zip(pred_tables, gold_tables)
        ]

    # ------------------------------------------------------------------
    # 1. Distribution
    # ------------------------------------------------------------------

    def get_error_distribution(self) -> Counter:
        """
        Return a Counter of how often each error type appears.

        A single table can contribute to multiple error types.
        """
        counter: Counter = Counter()
        for error_set in self._errors:
            for e in error_set:
                counter[e] += 1
        return counter

    def error_distribution_df(self) -> pd.DataFrame:
        """
        Return error distribution as a tidy DataFrame with counts and
        percentages.
        """
        dist = self.get_error_distribution()
        total = len(self._errors)
        rows = []
        for et in ALL_ERROR_TYPES:
            cnt = dist.get(et, 0)
            rows.append({"error_type": et, "count": cnt,
                         "pct": round(100 * cnt / total, 1)})
        return pd.DataFrame(rows).set_index("error_type")

    # ------------------------------------------------------------------
    # 2. Per-table annotation
    # ------------------------------------------------------------------

    def annotated_table(self, table_idx: int) -> Dict:
        """
        Return a dict describing the errors for a single table pair.

        Parameters
        ----------
        table_idx : int
            Index into pred_tables / gold_tables.

        Returns
        -------
        dict with keys: ``table_idx``, ``errors``, ``pred``, ``gold``,
        and (if scores_df provided) ``metric_scores``.
        """
        if table_idx < 0 or table_idx >= len(self.pred_tables):
            raise IndexError(f"table_idx {table_idx} out of range.")

        result = {
            "table_idx": table_idx,
            "errors": sorted(self._errors[table_idx]),
            "pred": self.pred_tables[table_idx],
            "gold": self.gold_tables[table_idx],
        }
        if self.scores_df is not None:
            result["metric_scores"] = {
                col: round(float(self.scores_df.iloc[table_idx][col]), 4)
                for col in self.scores_df.columns
            }
        return result

    # ------------------------------------------------------------------
    # 3. Metric sensitivity per error type
    # ------------------------------------------------------------------

    def metric_sensitivity_report(
        self,
        high_threshold: float = 0.6,
        low_threshold: float = 0.4,
    ) -> pd.DataFrame:
        """
        Build a table showing how sensitive each metric is to each error type.

        For each (metric, error_type) pair, the *sensitivity* is the fraction
        of tables with that error type where the metric score falls below
        ``low_threshold`` — i.e., the metric correctly flags the error.

        A **high** sensitivity value means the metric reliably detects that
        error type. A **low** value means the metric misses it.

        Parameters
        ----------
        high_threshold, low_threshold : float
            Normalised score thresholds used to decide "caught" vs "missed".

        Returns
        -------
        pd.DataFrame
            Rows = error types, columns = metrics, values = sensitivity
            (0–1, higher = better at catching that error).
        """
        if self.scores_df is None:
            raise ValueError(
                "scores_df must be provided to compute metric sensitivity."
            )

        # Normalise scores to [0, 1]
        normed = self.scores_df.copy()
        for col in normed.columns:
            mn, mx = normed[col].min(), normed[col].max()
            normed[col] = (normed[col] - mn) / (mx - mn + 1e-9)

        metrics = normed.columns.tolist()
        results = {}

        for et in ALL_ERROR_TYPES:
            if et == ErrorType.PERFECT:
                continue
            # Indices of tables that have this error
            error_indices = [
                i for i, es in enumerate(self._errors) if et in es
            ]
            if not error_indices:
                results[et] = {m: float("nan") for m in metrics}
                continue

            row = {}
            for metric in metrics:
                # "Caught" = score is low when error is present
                scores_with_error = normed[metric].iloc[error_indices]
                caught = (scores_with_error <= low_threshold).sum()
                sensitivity = caught / len(error_indices)
                row[metric] = round(float(sensitivity), 3)
            results[et] = row

        df = pd.DataFrame(results).T
        df.index.name = "error_type"
        return df

    # ------------------------------------------------------------------
    # 4. Summary report
    # ------------------------------------------------------------------

    def summary_report(self) -> str:
        lines = ["=" * 60, "ERROR TAXONOMY REPORT", "=" * 60, ""]
        dist = self.error_distribution_df()
        lines.append("Error Distribution:")
        lines.append(dist.to_string())

        if self.scores_df is not None:
            lines += ["", "Metric Sensitivity (fraction of errors caught):"]
            try:
                sens = self.metric_sensitivity_report()
                lines.append(sens.to_string())
            except Exception as exc:
                lines.append(f"  (could not compute: {exc})")

        return "\n".join(lines)
