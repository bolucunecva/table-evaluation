"""
metric_recommender.py
=====================
Recommend the best metric combination for a given evaluation context,
grounded in the empirical correlation results from TabXBench (Table 2 of
the Text2TabEval paper).

The recommendation logic uses a decision tree built from observed
patterns:
  - Alignment-agnostic metrics (EM, ChrF, BERTScore) score poorly on
    shuffled rows; TabXEval + H-Score handle this best.
  - RMSE is essential when numeric accuracy matters.
  - LLM-based metrics have the highest human correlation but require GPU.
  - For scientific domains with fixed schemas, ChrF + BERTScore + RMSE
    provide a fast, strong baseline.

Typical usage
-------------
>>> from text2tabeval.analysis.metric_recommender import MetricRecommender
>>> rec = MetricRecommender()
>>> result = rec.recommend(
...     compute_budget="low",
...     domain="general",
...     has_numeric_cols=True,
...     schema_is_fixed=False,
... )
>>> print(result["primary"])
>>> print(result["rationale"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Evidence base — grounded in Table 2 of the paper (TabXBench correlations)
# ---------------------------------------------------------------------------

# Spearman ρ from paper Table 2 (Qwen3-4B for LLM metrics)
_TABXBENCH_SPEARMAN = {
    "EM_cell":            0.125,
    "EM_column":          0.119,
    "ChrF_cell":          0.131,
    "ChrF_column":        0.116,
    "BERTScore_cell":     0.102,
    "BERTScore_column":   0.096,
    "Cell_F1":            0.204,
    "ROUGE-L":            0.208,
    "Levenshtein":        0.191,
    "CMT_cell":           0.073,
    "CMT_row":            0.135,
    "CMT_column":         0.079,
    "H-Score_content":    0.174,
    "H-Score_structure":  0.151,
    "P-Score_content":    0.072,
    "P-Score_structure":  0.070,
    "TabEval":            0.387,
    "TabXEval":           0.405,
}

# Group metadata
_METRIC_GROUPS = {
    "surface":   ["EM_cell", "EM_column", "ChrF_cell", "ChrF_column",
                  "ROUGE-L", "Levenshtein"],
    "embedding": ["BERTScore_cell", "BERTScore_column", "Cell_F1"],
    "alignment": ["CMT_cell", "CMT_row", "CMT_column",
                  "H-Score_content", "H-Score_structure"],
    "llm":       ["P-Score_content", "P-Score_structure", "TabEval", "TabXEval"],
    "numeric":   ["RMSE"],   # not in TabXBench but domain-critical
}

_COMPUTE_COST = {
    # (gpu_required, approx_seconds_per_100_tables)
    "EM_cell":           (False, 1),
    "ChrF_cell":         (False, 2),
    "ChrF_column":       (False, 2),
    "BERTScore_cell":    (True,  30),
    "BERTScore_column":  (True,  30),
    "Cell_F1":           (False, 2),
    "ROUGE-L":           (False, 1),
    "Levenshtein":       (False, 1),
    "H-Score_content":   (False, 5),
    "H-Score_structure": (False, 5),
    "CMT_row":           (False, 5),
    "P-Score_content":   (True,  120),
    "TabEval":           (True,  180),
    "TabXEval":          (True,  200),
    "RMSE":              (False, 1),
}


# ---------------------------------------------------------------------------
# Recommendation dataclass
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    primary: List[str]          = field(default_factory=list)
    secondary: List[str]        = field(default_factory=list)
    avoid: List[str]            = field(default_factory=list)
    rationale: str              = ""
    human_correlation_note: str = ""
    compute_note: str           = ""

    def pretty_print(self) -> None:
        print("=" * 60)
        print("METRIC RECOMMENDATION")
        print("=" * 60)
        print(f"Primary   : {', '.join(self.primary)}")
        print(f"Secondary : {', '.join(self.secondary)}")
        print(f"Avoid     : {', '.join(self.avoid)}")
        print(f"\nRationale : {self.rationale}")
        print(f"Human corr: {self.human_correlation_note}")
        print(f"Compute   : {self.compute_note}")


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class MetricRecommender:
    """
    Context-aware metric recommender for T2T evaluation.

    All recommendations are grounded in the TabXBench Spearman
    correlation results from Table 2 of the Text2TabEval paper.
    """

    def recommend(
        self,
        compute_budget: Literal["low", "medium", "high"] = "medium",
        domain: Literal["general", "scientific", "financial"] = "general",
        has_numeric_cols: bool = False,
        schema_is_fixed: bool = False,
        row_shuffle_likely: bool = False,
        human_judgements_available: bool = False,
    ) -> Recommendation:
        """
        Return a Recommendation for the given evaluation context.

        Parameters
        ----------
        compute_budget : "low" | "medium" | "high"
            "low"    — CPU only, fast (no transformers inference)
            "medium" — GPU available, small open-source LLMs OK (≤7B)
            "high"   — GPU + large/closed-source LLMs OK
        domain : "general" | "scientific" | "financial"
        has_numeric_cols : bool
            Whether tables contain numeric values that must be checked.
        schema_is_fixed : bool
            Whether column headers are fixed (same across all tables).
        row_shuffle_likely : bool
            Whether the model is known to shuffle rows.
        human_judgements_available : bool
            If True, recommend metrics with highest human correlation.

        Returns
        -------
        Recommendation dataclass
        """
        rec = Recommendation()
        rationale_parts = []

        # ---- Base: always start with lightweight reliable metrics ----
        rec.primary = ["ChrF_column", "H-Score_content", "Cell_F1"]
        rec.avoid   = ["EM_cell", "EM_column", "P-Score_content",
                       "P-Score_structure", "CMT_cell", "CMT_column"]

        rationale_parts.append(
            "ChrF_column and Cell_F1 provide fast, alignment-aware baselines "
            f"(ρ={_TABXBENCH_SPEARMAN['ChrF_column']:.3f} and "
            f"ρ={_TABXBENCH_SPEARMAN['Cell_F1']:.3f} on TabXBench)."
        )

        # ---- Compute budget ----
        if compute_budget == "high" or human_judgements_available:
            rec.primary.append("TabXEval")
            rec.secondary.append("TabEval")
            rationale_parts.append(
                f"TabXEval achieves the highest human correlation "
                f"(ρ={_TABXBENCH_SPEARMAN['TabXEval']:.3f}) and is recommended "
                "when GPU resources are available."
            )
            rec.compute_note = "Requires GPU; ~200s per 100 tables with a 4B model."

        elif compute_budget == "medium":
            rec.secondary.append("TabEval")
            rationale_parts.append(
                f"TabEval (ρ={_TABXBENCH_SPEARMAN['TabEval']:.3f}) provides "
                "strong LLM-based assessment at medium compute cost."
            )
            rec.compute_note = "TabEval requires GPU; use Qwen3-4B for efficiency."

        else:  # low
            rec.secondary.append("ROUGE-L")
            rec.secondary.append("Levenshtein")
            rationale_parts.append(
                "With low compute budget, ROUGE-L (ρ=0.208) and Levenshtein "
                "(ρ=0.191) offer the best human correlation without GPU."
            )
            rec.compute_note = "All primary metrics run on CPU in <5s per 100 tables."

        # ---- Numeric columns ----
        if has_numeric_cols:
            rec.primary.append("RMSE")
            rationale_parts.append(
                "RMSE is essential when tables contain numeric values — "
                "string-based metrics cannot detect numeric errors."
            )

        # ---- Row shuffle ----
        if row_shuffle_likely:
            if "H-Score_structure" not in rec.primary:
                rec.primary.append("H-Score_structure")
            rec.avoid.extend(["EM_cell", "ChrF_cell", "BERTScore_cell"])
            # deduplicate avoid
            rec.avoid = list(dict.fromkeys(rec.avoid))
            rationale_parts.append(
                "Row-shuffle errors inflate alignment-agnostic metrics (EM, ChrF, "
                "BERTScore). H-Score_structure and TabXEval are row-order-robust."
            )

        # ---- Fixed schema ----
        if schema_is_fixed:
            # No need for schema-checking metrics; cell-level is sufficient
            if "H-Score_structure" not in rec.primary:
                pass  # skip — schema evaluation not needed
            rationale_parts.append(
                "With fixed column schemas, schema-mismatch metrics are redundant; "
                "focus on cell-level and numeric accuracy."
            )

        # ---- Domain ----
        if domain == "scientific":
            if "RMSE" not in rec.primary:
                rec.primary.append("RMSE")
            rec.secondary.append("BERTScore_column")
            rationale_parts.append(
                "Scientific tables often have domain-specific terminology; "
                "BERTScore_column handles paraphrases better than exact-match."
            )

        elif domain == "financial":
            if "RMSE" not in rec.primary:
                rec.primary.append("RMSE")
            rationale_parts.append(
                "Financial tables are numeric-heavy; RMSE is critical."
            )

        # ---- Human correlation summary ----
        top_metrics_by_corr = sorted(
            _TABXBENCH_SPEARMAN.items(), key=lambda x: -x[1]
        )[:3]
        rec.human_correlation_note = (
            "Highest human correlation on TabXBench: "
            + ", ".join(
                f"{m} (ρ={r:.3f})" for m, r in top_metrics_by_corr
            )
        )

        # Deduplicate primary and secondary
        rec.primary   = list(dict.fromkeys(rec.primary))
        rec.secondary = list(dict.fromkeys(rec.secondary))

        rec.rationale = " ".join(rationale_parts)
        return rec

    # ------------------------------------------------------------------
    # Context-free "what is each metric best for" reference
    # ------------------------------------------------------------------

    def metric_profile(self, metric_name: str) -> Dict:
        """
        Return a profile dict for a single metric.

        Parameters
        ----------
        metric_name : str
            Must be a key in the TabXBench correlation table.

        Returns
        -------
        dict with keys: ``human_correlation``, ``gpu_required``,
        ``best_for``, ``limitations``.
        """
        profiles = {
            "TabXEval": {
                "human_correlation": _TABXBENCH_SPEARMAN["TabXEval"],
                "gpu_required": True,
                "best_for": "Overall quality with structural and semantic alignment",
                "limitations": "Requires LLM inference; slow on large datasets",
            },
            "TabEval": {
                "human_correlation": _TABXBENCH_SPEARMAN["TabEval"],
                "gpu_required": True,
                "best_for": "Semantic correctness with NLI backbone",
                "limitations": "Requires NLI model (roberta-large-mnli) + LLM",
            },
            "H-Score_content": {
                "human_correlation": _TABXBENCH_SPEARMAN["H-Score_content"],
                "gpu_required": False,
                "best_for": "Alignment-aware content evaluation without LLM",
                "limitations": "Does not assess numeric accuracy",
            },
            "ChrF_column": {
                "human_correlation": _TABXBENCH_SPEARMAN["ChrF_column"],
                "gpu_required": False,
                "best_for": "Fast character-level overlap at column granularity",
                "limitations": "Alignment-agnostic; misses row-shuffle errors",
            },
            "BERTScore_column": {
                "human_correlation": _TABXBENCH_SPEARMAN["BERTScore_column"],
                "gpu_required": True,
                "best_for": "Semantic similarity including paraphrases",
                "limitations": "Alignment-agnostic; high score even for wrong structure",
            },
            "RMSE": {
                "human_correlation": None,  # not in TabXBench
                "gpu_required": False,
                "best_for": "Numeric column accuracy",
                "limitations": "Only applicable to numeric columns; ignores text",
            },
            "EM_cell": {
                "human_correlation": _TABXBENCH_SPEARMAN["EM_cell"],
                "gpu_required": False,
                "best_for": "Strict exact-match correctness baseline",
                "limitations": "Very harsh; no partial credit; alignment-agnostic",
            },
        }
        if metric_name not in profiles:
            return {
                "human_correlation": _TABXBENCH_SPEARMAN.get(metric_name),
                "gpu_required": _COMPUTE_COST.get(metric_name, (None,))[0],
                "best_for": "See paper for details",
                "limitations": "Profile not yet defined",
            }
        return profiles[metric_name]

    def all_profiles(self) -> Dict[str, Dict]:
        """Return profiles for all metrics with known TabXBench correlations."""
        return {m: self.metric_profile(m) for m in _TABXBENCH_SPEARMAN}
