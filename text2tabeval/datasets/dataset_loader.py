import os
from typing import List, Dict

_BUILTIN_DATASETS = ["e2e", "rotowire", "wikitabletext"]
_DATASET_DIR = os.path.join(os.path.dirname(__file__))

def _dataset_path(dataset_name: str, split: str = "test") -> str:
    name = dataset_name.lower()
    if name not in _BUILTIN_DATASETS:
        raise ValueError(
            f"Unknown built-in dataset '{dataset_name}'. "
            f"Available: {_BUILTIN_DATASETS}. "
            f"For custom datasets, use load_pred_dataset(path)."
        )
    return os.path.join(_DATASET_DIR, name, f"{split}.data")

def parse_text_to_table(text) -> List[List[str]]:
    """
    Parse a Markdown-formatted table string into a 2-D list of cell values.

    Parameters
    ----------
    text : str or list
        Either a raw Markdown table string (rows separated by ``\\n`` or
        ``<NEWLINE>``), or a list of row strings (accepted for robustness).

    Returns
    -------
    List[List[str]]
        Rows of cell values, header separator rows removed.

    Raises
    ------
    TypeError
        If ``text`` is neither a str nor a list.
    """
    # ---- BUG FIX: the original code assumed str but received list ----
    if isinstance(text, list):
        text = "\n".join(text)
    elif not isinstance(text, str):
        raise TypeError(
            f"parse_text_to_table expects str or list, got {type(text).__name__}"
        )

    # Normalise <NEWLINE> tokens used in .data files
    text = text.replace("<NEWLINE>", "\n")

    rows = [r.strip() for r in text.strip().split("\n") if r.strip()]

    table = []
    for row in rows:
        # Skip Markdown separator rows like |---|---|
        if all(c in "-| :" for c in row):
            continue
        cells = [c.strip() for c in row.strip("|").split("|")]
        table.append(cells)

    return table
    
def validate_table_pair(
    pred_tables: List,
    gold_tables: List,
    context: str = "",
) -> None:
    """
    Raise informative errors if pred/gold lists are incompatible.

    Parameters
    ----------
    pred_tables, gold_tables : list
        Lists of parsed tables.
    context : str
        Name of the calling metric, shown in the error message.
    """
    prefix = f"[{context}] " if context else ""
    if not pred_tables:
        raise ValueError(f"{prefix}pred_tables is empty.")
    if not gold_tables:
        raise ValueError(f"{prefix}gold_tables is empty.")
    if len(pred_tables) != len(gold_tables):
        raise ValueError(
            f"{prefix}Length mismatch: "
            f"{len(pred_tables)} predicted vs {len(gold_tables)} gold tables."
        )

def _load_data_file(path: str) -> List[List[List[str]]]:
    """Read a .data file; each line → one table."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"Data file is empty: {path!r}")

    tables = [parse_text_to_table(line) for line in lines]
    return tables
    
def load_gold_dataset(
    dataset_name: str,
    split: str = "test",
) -> List[List[List[str]]]:
    """
    Load a built-in gold (reference) dataset.

    Parameters
    ----------
    dataset_name : str
        One of ``"e2e"``, ``"rotowire"``, ``"wikitabletext"``.
    split : str
        Dataset split — ``"test"`` (default), ``"train"``, or ``"dev"``.

    Returns
    -------
    List of parsed tables (each table is a list of row-lists).
    """
    path = _dataset_path(dataset_name, split)
    return _load_data_file(path)

def load_pred_dataset(path: str) -> List[List[List[str]]]:
    """
    Load predicted tables from a ``.data`` file.

    Parameters
    ----------
    path : str
        Path to a ``.data`` file where each line is one Markdown table
        (with ``<NEWLINE>`` separating rows within a table).

    Returns
    -------
    List of parsed tables.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path!r}")
    return _load_data_file(path)

