"""
metric_agreement.py
===================
Compute pairwise correlations between metrics, cluster them into agreement
groups, and surface table-level divergence cases.

Typical usage
-------------
>>> from text2tabeval.analysis.metric_agreement import MetricAgreement
>>> ma = MetricAgreement(scores_df)          # rows=tables, cols=metrics
>>> ma.correlation_matrix()                  # pandas DataFrame
>>> ma.plot_heatmap()                        # saves / shows matplotlib figure
>>> ma.get_divergence_cases(threshold=0.4)  # list of dicts
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


class MetricAgreement:
    """
    Analyse agreement and disagreement between evaluation metrics.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Shape ``(n_tables, n_metrics)``.  Each row is one table instance,
        each column is one metric's score.  ``NaN`` values are tolerated and
        dropped pairwise during correlation.
    """

    def __init__(self, scores_df: pd.DataFrame) -> None:
        if not isinstance(scores_df, pd.DataFrame):
            raise TypeError("scores_df must be a pandas DataFrame.")
        if scores_df.empty:
            raise ValueError("scores_df is empty.")
        self.scores = scores_df.copy()
        self._corr_matrix: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 1. Pairwise Spearman correlation matrix
    # ------------------------------------------------------------------

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Return Spearman correlation matrix between all metric columns.

        Returns
        -------
        pd.DataFrame
            Square matrix with metric names as index and columns.
        """
        metrics = self.scores.columns.tolist()
        n = len(metrics)
        mat = np.ones((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                xi = self.scores[metrics[i]]
                xj = self.scores[metrics[j]]
                # Drop rows where either metric is NaN
                mask = xi.notna() & xj.notna()
                if mask.sum() < 3:
                    mat[i, j] = mat[j, i] = np.nan
                    continue
                rho, _ = spearmanr(xi[mask], xj[mask])
                mat[i, j] = mat[j, i] = rho

        self._corr_matrix = pd.DataFrame(mat, index=metrics, columns=metrics)
        return self._corr_matrix

    # ------------------------------------------------------------------
    # 2. Cluster metrics into agreement groups
    # ------------------------------------------------------------------

    def metric_clusters(self, n_clusters: int = 3) -> Dict[int, List[str]]:
        """
        Cluster metrics by their correlation profile.

        Metrics that tend to agree with each other land in the same cluster.
        Useful for choosing a representative subset across diverse metric
        families (e.g., surface-form vs alignment vs LLM).

        Parameters
        ----------
        n_clusters : int
            Number of clusters (default 3: surface / alignment / LLM).

        Returns
        -------
        dict
            ``{cluster_id: [metric_name, ...]}``
        """
        corr = self.correlation_matrix().fillna(0).values
        # Distance = 1 − |correlation|
        dist = 1 - np.abs(corr)
        np.fill_diagonal(dist, 0)

        n_metrics = len(self.scores.columns)
        n_clusters = min(n_clusters, n_metrics)
        if n_clusters < 2:
            return {0: list(self.scores.columns)}

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist)

        clusters: Dict[int, List[str]] = {}
        for metric, label in zip(self.scores.columns, labels):
            clusters.setdefault(int(label), []).append(metric)
        return clusters

    # ------------------------------------------------------------------
    # 3. Divergence cases
    # ------------------------------------------------------------------

    def get_divergence_cases(
        self,
        threshold: float = 0.4,
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Find table instances where two metrics strongly disagree.

        A divergence is flagged when the *normalised* score of metric A is
        high (≥ 0.5 + threshold/2) while metric B is low (≤ 0.5 − threshold/2),
        or vice versa, after min-max scaling each metric to [0, 1].

        Parameters
        ----------
        threshold : float
            Minimum normalised score gap to flag (default 0.4).
        top_n : int
            Maximum number of divergence cases to return per metric pair.

        Returns
        -------
        list of dicts with keys:
            ``table_idx``, ``metric_high``, ``score_high``,
            ``metric_low``, ``score_low``, ``gap``
        """
        # Min-max normalise each metric to [0, 1]
        normed = self.scores.copy()
        for col in normed.columns:
            mn, mx = normed[col].min(), normed[col].max()
            if mx > mn:
                normed[col] = (normed[col] - mn) / (mx - mn)
            else:
                normed[col] = 0.5  # constant column → neutral

        metrics = normed.columns.tolist()
        cases = []

        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if j <= i:
                    continue
                diff = normed[m1] - normed[m2]
                for idx in diff.abs().nlargest(top_n).index:
                    gap = float(diff[idx])
                    if abs(gap) < threshold:
                        continue
                    high, low = (m1, m2) if gap > 0 else (m2, m1)
                    cases.append(
                        {
                            "table_idx": int(idx),
                            "metric_high": high,
                            "score_high": round(float(normed[high][idx]), 3),
                            "metric_low": low,
                            "score_low": round(float(normed[low][idx]), 3),
                            "gap": round(abs(gap), 3),
                        }
                    )

        # Sort by largest gap first, deduplicate by (table_idx, pair)
        seen = set()
        unique_cases = []
        for c in sorted(cases, key=lambda x: -x["gap"]):
            key = (c["table_idx"], frozenset([c["metric_high"], c["metric_low"]]))
            if key not in seen:
                seen.add(key)
                unique_cases.append(c)

        return unique_cases

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------

    def plot_heatmap(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """
        Plot a Spearman correlation heatmap across all metrics.

        Parameters
        ----------
        save_path : str, optional
            File path to save the figure (e.g., ``"heatmap.png"``).
        show : bool
            Whether to call ``plt.show()`` (default True).
        figsize : tuple
            Matplotlib figure size.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError(
                "matplotlib and seaborn are required for plot_heatmap(). "
                "Install with: pip install matplotlib seaborn"
            )

        corr = self.correlation_matrix()

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Spearman Correlation Between Evaluation Metrics", pad=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Summary report (plain text)
    # ------------------------------------------------------------------

    def summary_report(self) -> str:
        """
        Return a plain-text summary of metric agreement groups and top
        divergence cases. Useful for logging and paper tables.
        """
        lines = ["=" * 60, "METRIC AGREEMENT REPORT", "=" * 60]

        # Cluster summary
        clusters = self.metric_clusters()
        lines.append("\nMetric Clusters (by correlation profile):")
        for cid, members in clusters.items():
            lines.append(f"  Cluster {cid}: {', '.join(members)}")

        # Top divergences
        divs = self.get_divergence_cases(threshold=0.35, top_n=5)
        lines.append(f"\nTop Divergence Cases (threshold=0.35): {len(divs)} found")
        for d in divs[:10]:
            lines.append(
                f"  Table {d['table_idx']:>4}: "
                f"{d['metric_high']}={d['score_high']:.2f} vs "
                f"{d['metric_low']}={d['score_low']:.2f}  "
                f"(gap={d['gap']:.2f})"
            )

        return "\n".join(lines)
