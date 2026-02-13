import re
import numpy as np


def parse_text_to_table(text: str):
    """
    Parse a Markdown/LaTeX-style table string into a list of lists.
    Skips separator lines like | ---- | ---- | automatically.
    """
    rows = [r.strip() for r in text.strip().split('\n') if r.strip()]
    table = []

    for row in rows:
        row_content = row.strip('|').strip()
        # Skip separator lines containing only '-' or whitespace
        if all(c == '-' or c.isspace() for c in row_content.replace('|', '')):
            continue

        # Split cells and remove LaTeX $ formatting
        cells = [c.strip().strip('$') for c in row.strip('|').split('|')]
        table.append(cells)

    return table


def get_column_values(text: str, column_name: str):
    """
    Return ONLY the values from a specific column
    """
    table = parse_text_to_table(text)

    header = table[0]
    col_idx = header.index(column_name)
    return [row[col_idx] for row in table[1:]]


def value_to_float(value: str):
    """
    Convert LaTeX-style values like Value_{21} → 21.0
    Adjust this if your values follow a different rule.
    """
    try:
        return float(re.findall(r"\d+", value)[-1])
    except:
        return 0.0


def match_closest(a, b):
    """
    Match each value in a to the closest value in b
    """
    b = np.array(b)
    matches = []

    for x in a:
        idx = np.argmin(np.abs(b - x))
        matches.append(b[idx])

    return np.array(matches)


def evaluate_tables(hyp_text, tgt_text, column_name):
    """
    Compute RMSE between hypothesis and target tables for a given column
    """
    hyp_vals = [value_to_float(v) for v in get_column_values(hyp_text, column_name)]
    tgt_vals = [value_to_float(v) for v in get_column_values(tgt_text, column_name)]

    hyp_vals = np.array(hyp_vals)
    tgt_vals = np.array(tgt_vals)

    # Match smaller → larger for stability
    if len(hyp_vals) <= len(tgt_vals):
        matched_tgt = match_closest(hyp_vals, tgt_vals)
        matched_hyp = hyp_vals
    else:
        matched_hyp = match_closest(tgt_vals, hyp_vals)
        matched_tgt = tgt_vals

    return {"rmse": np.sqrt(np.mean((matched_hyp - matched_tgt) ** 2))}



# -------------------------
# Example Usage
# -------------------------

if __name__ == "__main__":
    hyp = """
    | $Column_1$ | $Column_2$ |
    |-----------------|----------------|
    | $Value_{11}$ | $Value_{12}$ |
    | $Value_{21}$ | $Value_{22}$ |
    """

    tgt = """
    | $Column_1$ | $Column_2$ |
    |-----------------|----------------|
    | $Value_{11}$ | $Value_{12}$ |
    | $Value_{21}$ | $Value_{22}$ |
    """

    rmse = evaluate_tables(hyp, tgt, "Column_1")
    print("RMSE:", rmse)
