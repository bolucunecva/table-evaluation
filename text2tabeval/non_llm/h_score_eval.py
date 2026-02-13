import numpy as np
import difflib
import Levenshtein

# ------------------
# Utilities
# ------------------
def normalize(x):
    """Lowercase, strip whitespace and $ symbols."""
    return str(x).strip().lower().strip("$")

def parse_md_table(md):
    """Parse Markdown/LaTeX table into list of rows with normalized strings."""
    if not md:
        return []

    rows = []
    for line in md.strip().split("\n"):
        line = line.strip()
        if all(c == '-' or c.isspace() for c in line.replace('|', '')):
            continue
        if "|" in line:
            cells = [normalize(c) for c in line.split("|")]
            # remove empty cells from ends
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            rows.append(cells)

    # Pad rows to max columns
    if rows:
        max_cols = max(len(r) for r in rows)
        for i in range(len(rows)):
            if len(rows[i]) < max_cols:
                rows[i] += [""] * (max_cols - len(rows[i]))

    return rows

def table_to_dicts(table):
    if not table:
        return []
    headers = table[0]
    return [dict(zip(headers, row)) for row in table[1:]]

def flatten(table):
    """Flatten table (list of dicts or lists) to list of strings."""
    if not table:
        return []
    if isinstance(table[0], dict):
        return [str(v) for row in table for v in row.values()]
    return [str(c) for row in table for c in row]

def lev_sim(a, b):
    a_str = " ".join(a)
    b_str = " ".join(b)
    if not b_str:
        return 0.0
    return 1 - Levenshtein.distance(a_str, b_str) / max(len(b_str), 1)

def diff_sim(a, b):
    a_str = " ".join(a)
    b_str = " ".join(b)
    return difflib.SequenceMatcher(None, a_str, b_str).ratio()

# ------------------
# H-Score (single table)
# ------------------
def h_score_single(gold_md, pred_md):
    structure, content = 0.0, 0.0
    gold_table = parse_md_table(gold_md)
    pred_table = parse_md_table(pred_md)

    if not gold_table or not pred_table:
        return 0.0, 0.0

    # Pad predicted table to match gold table shape
    max_rows = max(len(gold_table), len(pred_table))
    max_cols = max(len(gold_table[0]), len(pred_table[0]))
    
    def pad_table(table, rows, cols):
        # pad rows
        while len(table) < rows:
            table.append([""] * cols)
        # pad columns
        for i in range(len(table)):
            if len(table[i]) < cols:
                table[i] += [""] * (cols - len(table[i]))
        return table

    gold_table = pad_table(gold_table, max_rows, max_cols)
    pred_table = pad_table(pred_table, max_rows, max_cols)

    # Convert to dicts
    gold_dicts = table_to_dicts(gold_table)
    pred_dicts = table_to_dicts(pred_table)

    # Headers
    gold_cols, pred_cols = gold_table[0], pred_table[0]
    gold_data, pred_data = gold_dicts, pred_dicts

    # Structure similarity
    structure = np.mean([
        float(len(gold_data) == len(pred_data)),   # row count
        float(len(gold_cols) == len(pred_cols)),   # column count
        lev_sim(gold_cols, pred_cols),
        diff_sim(gold_cols, pred_cols),
    ])

    # Content similarity
    content = np.mean([
        lev_sim(flatten(gold_data), flatten(pred_data)),
        diff_sim(flatten(gold_data), flatten(pred_data)),
    ])

    return content, structure

# ------------------
# H-Score (aggregate)
# ------------------
def evaluate_tables(pred_tables, gold_tables):
    contents, structures = [], []

    for g_md, p_md in zip(gold_tables, pred_tables):
        c, s = h_score_single(g_md, p_md)
        contents.append(c)
        structures.append(s)

    return {
        "h_content_similarity": round(float(np.mean(contents)), 4),
        "h_structural_similarity": round(float(np.mean(structures)), 4),
    }

# ------------------
# Example usage
# ------------------
if __name__ == "__main__":
    gold_md = """
    | $obligation$ | $total$ | $<1yr$ | $1-3yr$ |
    | ---------- | ----- | ---- | ----- |
    | $debt$       | $100$   | $50$   | $50$    |
    | $lease$      | $200$   | $100$  | $100$   |
    """

    pred_md = """
    | $obligation$ | $total$ | $<1yr$ |
    | ---------- | ----- | ---- |
    | $debt$       | $100$   | $50$   |
    | $lease$      | $210$   | $110$  |
    """

    # Single table
    content, structure = h_score_single(gold_md, pred_md)
    print("Single table H-Score:")
    print("Content similarity:", content)
    print("Structure similarity:", structure)

    # Aggregate (list of tables)
    results = evaluate_tables([pred_md], [gold_md])
    print("\nAggregate H-Score:")
    print(results)
