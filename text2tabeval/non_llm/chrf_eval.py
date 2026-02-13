import numpy as np
from sacrebleu import sentence_chrf
from typing import List, Tuple

# -------------------------
# Table Parsing Utilities
# -------------------------

import numpy as np
from sacrebleu import sentence_chrf
from typing import Tuple

# -------------------------
# Table Parsing Utilities
# -------------------------

def parse_text_to_table(text: str):
    """
    Parse a LaTeX/Markdown-style table string into a list of lists.
    Skips separator lines like | ----- | ----- |
    """
    rows = [r.strip() for r in text.strip().split('\n') if r.strip()]
    table = []
    for row in rows:
        # Skip separator lines
        row_content = row.strip('|').strip()
        if all(c == '-' or c.isspace() for c in row_content.replace('|', '')):
            continue
        # Split cells and remove $ for LaTeX formatting
        cells = [c.strip().strip('$') for c in row.strip('|').split('|')]
        table.append(cells)
    return table

def is_empty_table(table, row_header: bool, col_header: bool) -> bool:
    """
    Check if table is empty.
    """
    if not table or len(table) == 0:
        return True
    if row_header and len(table) <= 1:
        return True
    if col_header and len(table[0]) <= 1:
        return True
    return False

# -------------------------
# Relation Extraction
# -------------------------

def parse_table_to_data(table, row_header: bool, col_header: bool):
    """
    Extract row headers, column headers, and cell relations from a list-of-lists table.
    Works with uneven rows and variable column counts.
    Returns:
        row_headers: set of row header strings
        col_headers: set of column header strings
        relations: set of tuples (row_header, col_header, value)
    """
    if not table or len(table) == 0:
        return set(), set(), set()

    # Column headers
    col_headers = table[0] if col_header else []
    if row_header and col_headers:
        col_headers = col_headers[1:]  # skip first cell if row header exists

    # Row headers
    row_headers = [row[0] for row in table[1:]] if row_header else []

    # Map column name to index for alignment
    col_index_map = {col_name: j for j, col_name in enumerate(table[0])} if col_header else {}

    # Relations
    relations = set()
    start_row = 1 if col_header else 0
    for i, row in enumerate(table[start_row:]):
        row_h = row[0] if row_header else None
        for j, col_name in enumerate(table[0] if col_header else range(len(row))):
            if row_header and col_header and j == 0:
                continue  # skip row-header cell in first column
            if j >= len(row):
                continue  # skip missing cells
            val = row[j]
            if val == "":
                continue
            rel = ()
            if row_header:
                rel += (row_h,)
            if col_header:
                rel += (col_name,)
            rel += (val,)
            relations.add(rel)

    return set(row_headers), set(col_headers), relations

# -------------------------
# chrF Metric Utilities
# -------------------------

metric_cache = dict()  # cache similarity results

def calc_data_similarity(tgt, pred) -> float:
    """
    Calculate chrF similarity between two strings or tuples recursively.
    """
    if isinstance(tgt, tuple):
        ret = 1.0
        for tt, pp in zip(tgt, pred):
            ret *= calc_data_similarity(tt, pp)
        return ret

    if (tgt, pred) in metric_cache:
        return metric_cache[(tgt, pred)]

    # chrF metric
    ret = sentence_chrf(pred, [tgt]).score / 100  # scale to 0..1
    metric_cache[(tgt, pred)] = ret
    return ret

def calc_similarity_matrix(tgt_data, pred_data):
    return [[calc_data_similarity(tgt, pred) for pred in pred_data] for tgt in tgt_data]

def metrics_by_sim(tgt_data, pred_data) -> Tuple[float, float, float]:
    sim = calc_similarity_matrix(tgt_data, pred_data)
    if not sim or not sim[0]:
        return 0.0, 0.0, 0.0
    # Convert to max across rows/columns for precision and recall
    prec = sum(max(col[i] for col in sim) for i in range(len(sim[0]))) / len(sim[0]) if len(sim[0]) > 0 else 0.0
    recall = sum(max(row) for row in sim) / len(sim) if len(sim) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return prec, recall, f1


# -------------------------
# Main Evaluation Function
# -------------------------

def evaluate_tables(hyp_table_str: str, tgt_table_str: str, row_header: bool = False, col_header: bool = False):
    hyp_table = parse_text_to_table(hyp_table_str)
    tgt_table = parse_text_to_table(tgt_table_str)

    if is_empty_table(tgt_table, row_header, col_header):
        print("Target table is empty. Skipping evaluation.")
        return

    hyp_row_headers, hyp_col_headers, hyp_relations = parse_table_to_data(hyp_table, row_header, col_header)
    tgt_row_headers, tgt_col_headers, tgt_relations = parse_table_to_data(tgt_table, row_header, col_header)

    # print("Hyp Row Headers:", hyp_row_headers, "Hyp Col Headers:", hyp_col_headers, "Hyp Relations:", hyp_relations)
    # print("Tgt Row Headers:", tgt_row_headers, "Tgt Col Headers:", tgt_col_headers, "Tgt Relations:", tgt_relations)
    # Row headers
    if row_header:
        p_row, r_row, f_row = metrics_by_sim(tgt_row_headers, hyp_row_headers)
        # print(f"Row header chrF: precision={p:.2f}, recall={r:.2f}, f1={f:.2f}")

    # Column headers
    if col_header:
        p_col, r_col, f_col = metrics_by_sim(tgt_col_headers, hyp_col_headers)
        # print(f"Column header chrF: precision={p:.2f}, recall={r:.2f}, f1={f:.2f}")

    # Non-header cells
    p_non_header, r_non_header, f_non_header = metrics_by_sim(tgt_relations, hyp_relations)
    # print(f"Non-header cells chrF: precision={p:.2f}, recall={r:.2f}, f1={f:.2f}")

    return {
        "row_header_precision": p_row if row_header else None,
        "row_header_recall": r_row if row_header else None,
        "row_header_f1": f_row if row_header else None,
        "col_header_precision": p_col if col_header else None,
        "col_header_recall": r_col if col_header else None,
        "col_header_f1": f_col if col_header else None,
        "non_header_precision": p_non_header,
        "non_header_recall": r_non_header,
        "non_header_f1": f_non_header
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

    evaluate_tables(hyp, tgt, row_header=True, col_header=True)
