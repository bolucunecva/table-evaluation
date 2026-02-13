import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import Levenshtein

# ---------------------------
# Utilities
# ---------------------------
def normalize_cell(value):
    """Normalize table cells for comparison, remove $ symbols."""
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip().lower().strip("$")
    return str(value).strip()

def markdown_to_df(markdown_table):
    """Convert a Markdown/LaTeX table string into a DataFrame."""
    lines = [l.strip() for l in markdown_table.strip().split('\n') if l.strip()]
    
    if not lines:
        return pd.DataFrame()

    # Helper to detect separator lines like |----|----|
    def is_separator(line):
        content = line.replace('|', '').strip()
        return content and all(c in '-: ' for c in content)

    # Header is first non-separator line
    header_line = lines[0]
    header = [normalize_cell(col) for col in header_line.split('|') if col.strip()]

    data = []
    for line in lines[1:]:
        if is_separator(line):
            continue
        row = [normalize_cell(col) for col in line.split('|') if col.strip()]
        data.append(row)

    if not data:
        return pd.DataFrame(columns=header)

    num_cols = max(len(header), max(len(row) for row in data))
    header += [""] * (num_cols - len(header))
    data = [row + [""] * (num_cols - len(row)) for row in data]

    return pd.DataFrame(np.array(data, dtype=object), columns=header)

# ---------------------------
# Table Evaluation
# ---------------------------
def evaluate_table(gold_md, pred_md, rouge=None):
    """Evaluate a single Markdown table pair."""
    if rouge is None:
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    df_gt = markdown_to_df(gold_md)
    df_pred = markdown_to_df(pred_md)

    # Pad tables to same shape
    max_rows = max(df_gt.shape[0], df_pred.shape[0])
    max_cols = max(df_gt.shape[1], df_pred.shape[1])
    df_gt_pad = pd.DataFrame(np.full((max_rows, max_cols), ""), columns=range(max_cols))
    df_pred_pad = pd.DataFrame(np.full((max_rows, max_cols), ""), columns=range(max_cols))
    df_gt_pad.iloc[:df_gt.shape[0], :df_gt.shape[1]] = df_gt.values
    df_pred_pad.iloc[:df_pred.shape[0], :df_pred.shape[1]] = df_pred.values

    # Cell-level metrics
    tp_cell = np.sum(df_gt_pad.values == df_pred_pad.values)
    fp_cell = np.sum((df_pred_pad.values != df_gt_pad.values) & (df_pred_pad.values != ""))
    fn_cell = np.sum((df_gt_pad.values != df_pred_pad.values) & (df_gt_pad.values != ""))
    precision_cell = tp_cell / (tp_cell + fp_cell) if tp_cell + fp_cell > 0 else 0
    recall_cell = tp_cell / (tp_cell + fn_cell) if tp_cell + fn_cell > 0 else 0
    f1_cell = 2 * precision_cell * recall_cell / (precision_cell + recall_cell) if precision_cell + recall_cell > 0 else 0

    # Table-level accuracy (cells + headers)
    gt_headers = [normalize_cell(c) for c in df_gt.columns]
    pred_headers = [normalize_cell(c) for c in df_pred.columns]
    max_cols = max(len(gt_headers), len(pred_headers))
    gt_headers_pad = gt_headers + [""]*(max_cols - len(gt_headers))
    pred_headers_pad = pred_headers + [""]*(max_cols - len(pred_headers))
    headers_match = np.sum(np.array(gt_headers_pad) == np.array(pred_headers_pad))
    cells_match = tp_cell
    table_accuracy = (cells_match + headers_match) / (df_gt_pad.size + len(gt_headers_pad))

    # Sequence-level metrics
    def table_to_string(df):
        return "\n".join([" | ".join(map(str, row)) for row in df.values])
    gt_str = table_to_string(df_gt_pad)
    pred_str = table_to_string(df_pred_pad)
    rouge_l = rouge.score(gt_str, pred_str)['rougeL'].fmeasure
    lev_ratio = Levenshtein.ratio(gt_str, pred_str)

    return {
        'cell_precision': precision_cell,
        'cell_recall': recall_cell,
        'cell_f1': f1_cell,
        'table_accuracy': table_accuracy,
        'rouge_l': rouge_l,
        'levenshtein_ratio': lev_ratio
    }

def evaluate_tables(pred_tables, gold_tables):
    """Evaluate multiple Markdown tables and return aggregated results."""
    assert len(pred_tables) == len(gold_tables), "Prediction and gold counts must match"
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = [evaluate_table(g, p, rouge) for g, p in zip(gold_tables, pred_tables)]

    # Aggregate metrics
    agg = {}
    for key in results[0].keys():
        agg[key] = np.mean([r[key] for r in results])
    return agg

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    hyp = """
    | $Column_1$ | $Column_2$ |
    | $Value_{11}$ | $Value_{12}$ |
    | $Value_{21}$ | $Value_{22}$ |
    """

    tgt = """
    | $Column_1$ | $Column_2$ |
    | $Value_{11}$ | $Value_{12}$ |
    | $Value_{21}$ | $Value_{22}$ |
    """

    result = evaluate_tables([hyp], [tgt])
    print(result)
