import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import Levenshtein

def normalize_cell(value):
    """Normalize table cells for exact comparison."""
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip().lower()
    return str(value)

class HierarchicalTableEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate(self, df_gt, df_pred):
        # Ensure same shape by padding missing rows/cols
        max_rows = max(df_gt.shape[0], df_pred.shape[0])
        max_cols = max(df_gt.shape[1], df_pred.shape[1])
        df_gt_pad = pd.DataFrame(np.full((max_rows, max_cols), ""), columns=range(max_cols))
        df_pred_pad = pd.DataFrame(np.full((max_rows, max_cols), ""), columns=range(max_cols))
        df_gt_pad.iloc[:df_gt.shape[0], :df_gt.shape[1]] = df_gt.values
        df_pred_pad.iloc[:df_pred.shape[0], :df_pred.shape[1]] = df_pred.values

        # Normalize
        df_gt_norm = df_gt_pad.applymap(normalize_cell)
        df_pred_norm = df_pred_pad.applymap(normalize_cell)

        # === Cell-Level ===
        tp_cell = np.sum(df_gt_norm.values == df_pred_norm.values)
        total_cells = df_gt_norm.size
        fp_cell = np.sum((df_pred_norm.values != df_gt_norm.values) & (df_pred_norm.values != ""))
        fn_cell = np.sum((df_gt_norm.values != df_pred_norm.values) & (df_gt_norm.values != ""))

        precision_cell = tp_cell / (tp_cell + fp_cell) if tp_cell + fp_cell > 0 else 0
        recall_cell = tp_cell / (tp_cell + fn_cell) if tp_cell + fn_cell > 0 else 0
        f1_cell = 2 * precision_cell * recall_cell / (precision_cell + recall_cell) if precision_cell + recall_cell > 0 else 0

        # === Row-Level ===
        tp_row = 0
        for i in range(df_gt_norm.shape[0]):
            if any((df_gt_norm.iloc[i].values == df_pred_norm.iloc[j].values).all() for j in range(df_pred_norm.shape[0])):
                tp_row += 1
        fp_row = df_pred_norm.shape[0] - tp_row
        fn_row = df_gt_norm.shape[0] - tp_row
        precision_row = tp_row / (tp_row + fp_row) if tp_row + fp_row > 0 else 0
        recall_row = tp_row / (tp_row + fn_row) if tp_row + fn_row > 0 else 0
        f1_row = 2 * precision_row * recall_row / (precision_row + recall_row) if precision_row + recall_row > 0 else 0

        # === Table-Level Accuracy ===
        table_match = int((df_gt_norm.values == df_pred_norm.values).all())

        # === Levenshtein Ratio & ROUGE-L ===
        # Convert table to minimal markdown string for sequence-based metrics
        def table_to_string(df):
            return "\n".join([" | ".join(row) for row in df.values])

        gt_str = table_to_string(df_gt_norm)
        pred_str = table_to_string(df_pred_norm)

        rouge_l = self.rouge.score(gt_str, pred_str)['rougeL'].fmeasure
        lev_ratio = Levenshtein.ratio(gt_str, pred_str)

        results = {
            'cell_precision': precision_cell,
            'cell_recall': recall_cell,
            'cell_f1': f1_cell,
            'row_precision': precision_row,
            'row_recall': recall_row,
            'row_f1': f1_row,
            'table_accuracy': table_match,
            'rouge_l': rouge_l,
            'levenshtein_ratio': lev_ratio
        }

        return results