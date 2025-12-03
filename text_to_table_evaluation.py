import pandas as pd
import numpy as np
import tqdm
import bert_score
from sacrebleu import sentence_chrf

from table_utils import extract_table_by_name, parse_text_to_table, is_empty_table

# Global variables
bert_scorer = None
metric_cache = dict()


def normalize_table(table):
    table = np.where(pd.isna(table), "", table)
    table = table.astype(str)
    table = np.char.strip(table)
    return table


def parse_table_element_to_relation(table, i, j, row_header, col_header):
    assert row_header or col_header
    relation = []
    if row_header:
        assert j > 0
        relation.append(table[i][0])
    if col_header:
        assert i > 0
        relation.append(table[0][j])
    relation.append(table[i][j])
    return tuple(relation)


def parse_table_to_data(table, row_header, col_header):
    if is_empty_table(table, row_header, col_header):
        return set(), set(), set()

    row_headers = list(table[:, 0]) if row_header else []
    col_headers = list(table[0, :]) if col_header else []
    if row_header and col_header and table[0, 0] == "":
        row_headers = row_headers[1:]
        col_headers = col_headers[1:]

    row, col = table.shape
    relations = []
    for i in range(1 if col_header else 0, row):
        for j in range(1 if row_header else 0, col):
            relations.append(parse_table_element_to_relation(table, i, j, row_header, col_header))
    return set(row_headers), set(col_headers), set(relations)


def calc_similarity_matrix(tgt_data, pred_data, metric):
    global bert_scorer

    def calc_data_similarity(tgt, pred):
        if isinstance(tgt, tuple):
            ret = 1.0
            for tt, pp in zip(tgt, pred):
                ret *= calc_data_similarity(tt, pp)
            return ret

        if (tgt, pred) in metric_cache:
            return metric_cache[(tgt, pred)]

        if metric == "E":
            ret = int(tgt == pred)
        elif metric == "c":
            ret = sentence_chrf(pred, [tgt]).score / 100
        elif metric == "BS-scaled":
            if bert_scorer is None:
                bert_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
            ret = bert_scorer.score([pred], [tgt])[2].item()
            ret = max(min(ret, 1.0), 0.0)
        else:
            raise ValueError(f"Unknown metric {metric}")

        metric_cache[(tgt, pred)] = ret
        return ret

    return np.array([[calc_data_similarity(tgt, pred) for pred in pred_data] for tgt in tgt_data], dtype=float)


def metrics_by_sim(tgt_data, pred_data, metric):
    sim = calc_similarity_matrix(tgt_data, pred_data, metric)
    prec = np.mean(np.max(sim, axis=0)) if pred_data else 0.0
    recall = np.mean(np.max(sim, axis=1)) if tgt_data else 0.0
    f1 = 0.0 if prec + recall == 0 else 2 * prec * recall / (prec + recall)
    return prec, recall, f1


def evaluate_tables(hyp_df: pd.DataFrame, tgt_df: pd.DataFrame,
                    row_header: bool = True, col_header: bool = True, metric: str = "E"):
    """
    Evaluate predicted tables against ground truth tables.

    Args:
        hyp_df: Predicted tables (must include 'ID' column)
        tgt_df: Ground truth tables (must include 'ID' column)
        row_header: Evaluate row headers
        col_header: Evaluate column headers
        metric: "E" | "c" | "BS-scaled"

    Returns:
        results: dict of per-paper metrics
        dataset_metrics: dict of dataset-level averages
    """
    # Check ID column
    if "ID" not in tgt_df.columns or "ID" not in hyp_df.columns:
        raise ValueError("Both DataFrames must have 'ID' column for grouping.")

    paper_ids = sorted(tgt_df['ID'].unique())
    results = {}

    all_row_p, all_row_r, all_row_f = [], [], []
    all_col_p, all_col_r, all_col_f = [], [], []
    all_rel_p, all_rel_r, all_rel_f = [], [], []

    for pid in tqdm.tqdm(paper_ids, desc="Evaluating papers"):
        hyp_sub = hyp_df[hyp_df['ID'] == pid].drop(columns=['ID']).to_numpy(dtype=str)
        tgt_sub = tgt_df[tgt_df['ID'] == pid].drop(columns=['ID']).to_numpy(dtype=str)

        hyp_sub = normalize_table(hyp_sub)
        tgt_sub = normalize_table(tgt_sub)

        hyp_rows, hyp_cols, hyp_rel = parse_table_to_data(hyp_sub, row_header, col_header)
        tgt_rows, tgt_cols, tgt_rel = parse_table_to_data(tgt_sub, row_header, col_header)

        paper_metrics = {}
        if row_header:
            paper_metrics['row_p'], paper_metrics['row_r'], paper_metrics['row_f'] = metrics_by_sim(tgt_rows, hyp_rows, metric)
            all_row_p.append(paper_metrics['row_p'])
            all_row_r.append(paper_metrics['row_r'])
            all_row_f.append(paper_metrics['row_f'])
        if col_header:
            paper_metrics['col_p'], paper_metrics['col_r'], paper_metrics['col_f'] = metrics_by_sim(tgt_cols, hyp_cols, metric)
            all_col_p.append(paper_metrics['col_p'])
            all_col_r.append(paper_metrics['col_r'])
            all_col_f.append(paper_metrics['col_f'])
        paper_metrics['rel_p'], paper_metrics['rel_r'], paper_metrics['rel_f'] = metrics_by_sim(tgt_rel, hyp_rel, metric)
        all_rel_p.append(paper_metrics['rel_p'])
        all_rel_r.append(paper_metrics['rel_r'])
        all_rel_f.append(paper_metrics['rel_f'])

        results[pid] = paper_metrics

    dataset_metrics = {
        "row_p": np.mean(all_row_p) if all_row_p else 0.0,
        "row_r": np.mean(all_row_r) if all_row_r else 0.0,
        "row_f": np.mean(all_row_f) if all_row_f else 0.0,
        "col_p": np.mean(all_col_p) if all_col_p else 0.0,
        "col_r": np.mean(all_col_r) if all_col_r else 0.0,
        "col_f": np.mean(all_col_f) if all_col_f else 0.0,
        "rel_p": np.mean(all_rel_p) if all_rel_p else 0.0,
        "rel_r": np.mean(all_rel_r) if all_rel_r else 0.0,
        "rel_f": np.mean(all_rel_f) if all_rel_f else 0.0,
    }
    
    return {
        "row_p": np.mean(all_row_p) if all_row_p else 0.0,
        "row_r": np.mean(all_row_r) if all_row_r else 0.0,
        "row_f": np.mean(all_row_f) if all_row_f else 0.0,
        "col_p": np.mean(all_col_p) if all_col_p else 0.0,
        "col_r": np.mean(all_col_r) if all_col_r else 0.0,
        "col_f": np.mean(all_col_f) if all_col_f else 0.0,
        "rel_p": np.mean(all_rel_p) if all_rel_p else 0.0,
        "rel_r": np.mean(all_rel_r) if all_rel_r else 0.0,
        "rel_f": np.mean(all_rel_f) if all_rel_f else 0.0,
    }
        

    #return results, dataset_metrics