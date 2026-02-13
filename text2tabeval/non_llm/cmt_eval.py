import numpy as np
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
import pandas as pd

# -------------------------
# CMT-style Table Evaluation for Markdown/LaTeX tables
# -------------------------

def normalize_name(name):
    """Lowercase, remove whitespace and $ symbols."""
    if not isinstance(name, str):
        name = str(name)
    return name.lower().replace(" ", "").replace("$", "")

def token_overlap_score(str1, str2):
    """Compute token overlap ratio between two strings."""
    tokens1 = set(str1.split())
    tokens2 = set(str2.split())
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def composite_similarity(str1, str2):
    """60% token overlap + 40% SequenceMatcher ratio."""
    str1_norm = normalize_name(str1)
    str2_norm = normalize_name(str2)
    tok_score = token_overlap_score(str1_norm, str2_norm)
    seq_score = SequenceMatcher(None, str1_norm, str2_norm).ratio()
    return 0.6 * tok_score + 0.4 * seq_score

def align_entities(pred_entities, gold_entities, threshold=0.35):
    """Align predicted rows to gold rows using Hungarian algorithm."""
    if not pred_entities or not gold_entities:
        return {}
    sim_matrix = np.zeros((len(pred_entities), len(gold_entities)))
    for i, pred in enumerate(pred_entities):
        for j, gold in enumerate(gold_entities):
            sim_matrix[i, j] = composite_similarity(" ".join(pred), " ".join(gold))
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    mapping = {r: c for r, c in zip(row_ind, col_ind) if sim_matrix[r, c] >= threshold}
    return mapping

def parse_md_table(markdown_table):
    """Convert Markdown/LaTeX table string into a DataFrame with normalized cells."""
    lines = [line.strip() for line in markdown_table.strip().split("\n") if line.strip()]
    # Skip separator lines (---)
    lines = [line for line in lines if not all(c == '-' or c.isspace() or c == '|' for c in line)]
    if not lines:
        return pd.DataFrame()
    
    # Header
    header = [normalize_name(col.strip()) for col in lines[0].split("|") if col.strip()]
    # Data rows
    data = [
        [normalize_name(col.strip()) for col in line.split("|") if col.strip()]
        for line in lines[1:]
    ]
    if not data:
        data = []
    num_cols = max(len(header), max(len(row) for row in data) if data else 0)
    header += [""] * (num_cols - len(header))
    normalized_data = [row + [""]*(num_cols - len(row)) for row in data]
    df = pd.DataFrame(np.array(normalized_data, dtype=object), columns=header)
    return df

def compute_col_accuracy(pred_table, gold_table, threshold=0.35):
    """Compute column-level alignment accuracy (normalized 0..1)."""
    pred_cols = pred_table.columns.tolist()
    gold_cols = gold_table.columns.tolist()
    
    if not pred_cols or not gold_cols:
        return 0.0

    # Similarity matrix
    sim_matrix = np.zeros((len(pred_cols), len(gold_cols)))
    for i, p in enumerate(pred_cols):
        for j, g in enumerate(gold_cols):
            sim_matrix[i, j] = composite_similarity(p, g)

    # Hungarian alignment
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    matched = sum(sim_matrix[r, c] >= threshold for r, c in zip(row_ind, col_ind))
    
    # Normalize by number of gold columns
    col_acc = matched / len(gold_cols)
    return col_acc

def compute_cmt_metrics_md(pred_md, gold_md):
    """
    Compute CMT metrics for a single Markdown/LaTeX table.
    Returns dict with cell, row, and column accuracy.
    """
    pred_table = parse_md_table(pred_md)
    gold_table = parse_md_table(gold_md)

    pred_rows = pred_table.values.tolist()
    gold_rows = gold_table.values.tolist()

    # Align rows
    mapping = align_entities(pred_rows, gold_rows)

    correct_cells = 0
    row_correct = 0

    for pred_idx, gold_idx in mapping.items():
        pred_row = pred_rows[pred_idx]
        gold_row = gold_rows[gold_idx]

        if pred_row == gold_row:
            row_correct += 1

        for item in pred_row:
            if item in gold_row:
                correct_cells += 1

    total_cells = len(gold_rows) * len(gold_rows[0]) if gold_rows else 1
    cell_acc = correct_cells / total_cells
    row_acc = row_correct / len(gold_rows) if gold_rows else 0.0
    col_acc = compute_col_accuracy(pred_table, gold_table)  # proper column alignment

    return {
        "cell_accuracy": cell_acc,
        "row_accuracy": row_acc,
        "col_accuracy": col_acc
    }

def evaluate_tables(pred_md_list, gold_md_list):
    """Evaluate multiple Markdown/LaTeX tables and aggregate results."""
    all_cell = []
    all_row = []
    all_col = []

    for pred_md, gold_md in zip(pred_md_list, gold_md_list):
        metrics = compute_cmt_metrics_md(pred_md, gold_md)
        all_cell.append(metrics["cell_accuracy"])
        all_row.append(metrics["row_accuracy"])
        all_col.append(metrics["col_accuracy"])

    return {
        "cell_accuracy": np.mean(all_cell),
        "row_accuracy": np.mean(all_row),
        "col_accuracy": np.mean(all_col)
    }

# -------------------------
# Example Usage
# -------------------------
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
