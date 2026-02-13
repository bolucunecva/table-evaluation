from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import re
from ..utils.comparison_utils import *
from ..utils.fuzzy_table_matching import *

def take_final_table(text: str) -> str:

    # Extract everything after "The Final Aligned Table:"
    match = re.search(r"The Final Aligned Table:\s*(\|.*)", text, re.S)

    if match:
        final_table = match.group(1).strip()
        return final_table
    else:
        return ''
    
def build_prompt(system_prompt, user_text):
    if system_prompt:
        return f"""System:
{system_prompt}

User:
{user_text}

Assistant:
"""
    else:
        return f"""User:
{user_text}

Assistant:
"""
    
def evaluate_tables(
    perturbed_table,
    gold_table,
    llm_local,
    align_prompt=None,
    compare_prompt=None,
    allowed_data_types=None,
):
    """
    Main evaluation entry point.

    Usage:
    tabxeval.evaluate_tables(perturbed, gold, llm_local=llm)
    """

    if llm_local is None:
        raise ValueError("llm_local must be provided")

    if allowed_data_types is None:
        allowed_data_types = [
            "Numerical", "String", "Bool",
            "Date", "List", "Time", "Others"
        ]


    # -------- Step 1: Partial alignment --------
    partial_alignment = merge_tables_fuzzy(
        perturbed_table, gold_table
    )[0]

    prompt = (
        f"Align the following tables:\n\n"
        f"{perturbed_table}\n\n"
        f"{gold_table}\n\n"
    )

    if partial_alignment is not None:
        prompt += f"Partially Aligned Table:\n{partial_alignment}"
    
    if align_prompt is None:
        prompt_path = Path(__file__).parent.parent / "prompts" / "tabalign.txt"
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    if compare_prompt is None:
        prompt_path = Path(__file__).parent.parent / "prompts" / "tabcompare.txt"
    compare_prompt = Path(prompt_path).read_text(encoding="utf-8")


    align_prompt = prompt_template.format(
            partial_table=partial_alignment, table1=perturbed_table, table2=gold_table
        )


    alignment = take_final_table(llm_local.generate(align_prompt))

    # -------- Step 2: Parse alignment --------
    df_replaced, df_compare, df_wo_em = compare(alignment)

    # -------- Step 3: Comparison tuples --------
    table_md = df_wo_em.to_markdown(index=False)

    compare_prompt = build_prompt(table_md, compare_prompt)
    comparison_tuples = llm_local.generate(compare_prompt)


    parsed_tuples = table_to_dict_list_comparison(comparison_tuples)
    parsed_tuples = [
        {k: None if v is None else parse_string(v) for k, v in row.items()}
        for row in parsed_tuples
    ]
    # print(parsed_tuples)
    # -------- Step 4: Statistics --------
    bundle = [{
        "alignment": alignment,
        "comparison_tuples_parsed": parsed_tuples
    }]

    bundle = get_partial_cells_stats(bundle, allowed_data_types)[0]

    delta_stats = make_delta_stats_table(
        bundle['delta'], bundle['type_counts']
    )

    ei_mi_table = pd.DataFrame(bundle['ei_mi_table'])


    row_col_stats = get_row_column_statistics(
        ei_mi_table, df_compare
    )

    cell_stats = create_summary_table_from_df(
        ei_mi_table,
        allowed_data_types,
        default_categories=["EI", "MI", "Partial"],
        partial=bundle['empty_cells']['Partial']
    )


    return {
        "alignment": alignment,
        "df_compare": df_compare,
        "df_wo_em": df_wo_em,
        "comparison_tuples_parsed": parsed_tuples,
        "delta_stats": delta_stats,
        "row_col_statistics": row_col_stats,
        "cell_stats": cell_stats
    }