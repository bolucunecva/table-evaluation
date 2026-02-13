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
        row_content = row.strip('|').strip()
        # Skip separator lines
        if all(c == '-' or c.isspace() for c in row_content.replace('|', '')):
            continue
        # Split cells and remove $ for LaTeX formatting
        cells = [c.strip().strip('$') for c in row.strip('|').split('|')]
        table.append(cells)
    return table


def is_empty_table(table, row_header: bool, col_header: bool) -> bool:
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
    if not table or len(table) == 0:
        return set(), set(), set()

    col_headers = table[0] if col_header else []
    if row_header and col_headers:
        col_headers = col_headers[1:]

    row_headers = [row[0] for row in table[1:]] if row_header else []

    relations = set()
    start_row = 1 if col_header else 0

    for row in table[start_row:]:
        row_h = row[0] if row_header else None
        for j, col_name in enumerate(table[0] if col_header else range(len(row))):
            if row_header and col_header and j == 0:
                continue
            if j >= len(row):
                continue

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
# Exact Match Utilities
# -------------------------

def calc_data_similarity(tgt, pred) -> float:
    """
    Exact match similarity.
    Returns 1.0 if tgt == pred, else 0.0.
    Works recursively for tuples.
    """
    if isinstance(tgt, tuple) and isinstance(pred, tuple):
        if len(tgt) != len(pred):
            return 0.0
        return 1.0 if all(
            calc_data_similarity(t, p) == 1.0 for t, p in zip(tgt, pred)
        ) else 0.0

    return 1.0 if tgt == pred else 0.0


def calc_similarity_matrix(tgt_data, pred_data):
    return [[calc_data_similarity(tgt, pred) for pred in pred_data]
            for tgt in tgt_data]


def metrics_by_sim(tgt_data, pred_data) -> Tuple[float, float, float]:
    if not tgt_data or not pred_data:
        return 0.0, 0.0, 0.0

    sim = calc_similarity_matrix(tgt_data, pred_data)

    prec = (
        sum(max(col[i] for col in sim) for i in range(len(sim[0])))
        / len(sim[0])
        if len(sim[0]) > 0 else 0.0
    )

    recall = (
        sum(max(row) for row in sim) / len(sim)
        if len(sim) > 0 else 0.0
    )

    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return prec, recall, f1

# -------------------------
# Main Evaluation Function
# -------------------------

def evaluate_tables(
    hyp_table_str: str,
    tgt_table_str: str,
    row_header: bool = False,
    col_header: bool = False
):
    hyp_table = parse_text_to_table(hyp_table_str)
    tgt_table = parse_text_to_table(tgt_table_str)

    if is_empty_table(tgt_table, row_header, col_header):
        print("Target table is empty. Skipping evaluation.")
        return

    hyp_row_headers, hyp_col_headers, hyp_relations = parse_table_to_data(
        hyp_table, row_header, col_header
    )
    tgt_row_headers, tgt_col_headers, tgt_relations = parse_table_to_data(
        tgt_table, row_header, col_header
    )

    if row_header:
        p_row, r_row, f_row = metrics_by_sim(tgt_row_headers, hyp_row_headers)
    else:
        p_row = r_row = f_row = None

    if col_header:
        p_col, r_col, f_col = metrics_by_sim(tgt_col_headers, hyp_col_headers)
    else:
        p_col = r_col = f_col = None

    p_non, r_non, f_non = metrics_by_sim(tgt_relations, hyp_relations)

    return {
        "row_header_precision": p_row,
        "row_header_recall": r_row,
        "row_header_f1": f_row,
        "col_header_precision": p_col,
        "col_header_recall": r_col,
        "col_header_f1": f_col,
        "non_header_precision": p_non,
        "non_header_recall": r_non,
        "non_header_f1": f_non,
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

    results = evaluate_tables(hyp, tgt, row_header=True, col_header=True)
    print(results)
