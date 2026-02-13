import os

def parse_md_table_line(line):
    """Convert a single line of MD table with <NEWLINE> into proper Markdown table string."""
    rows = line.split("<NEWLINE>")
    rows = [row.strip() for row in rows if row.strip()]
    table_md = "\n".join(rows)
    return table_md

def load_md_file(file_path):
    """Load a file where each line is a Markdown table."""
    tables = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tables.append(parse_md_table_line(line))
    return tables

def load_gold_dataset(name, base_path=os.path.join(os.path.dirname(__file__))):
    """
    Load gold/reference tables for a dataset.
    name: dataset folder name (e.g., 'restaurant')
    """
    dataset_folder = os.path.join(base_path, name)
    gold_path = os.path.join(dataset_folder, "test.data")
    gold_tables = load_md_file(gold_path)
    return gold_tables

def load_pred_dataset(file_path):
    """
    Load user-provided predictions (same format as gold).
    """
    return load_md_file(file_path)
