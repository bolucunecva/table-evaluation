import pandas as pd
from unidecode import unidecode
from sentence_transformers import SentenceTransformer, util

class TableEvaluator:
    """
    Evaluate table extraction using key matching, non-key cell matching, and optional semantic similarity.
    """
    def __init__(self, use_semantic=True, semantic_model_path='../models/all-MiniLM-L6-v2'):
        self.use_semantic = use_semantic
        if self.use_semantic:
            print("Loading semantic embedding model...")
            self.model = SentenceTransformer(semantic_model_path)
        else:
            self.model = None

    def normalize_value(self, value, is_date=False):
        """Normalize cell value: string/number/date."""
        if pd.isna(value):
            return ''

        if is_date:
            try:
                return pd.to_datetime(value)
            except:
                return value

        if isinstance(value, str):
            v = value.strip().lower()
            if v in ('none', 'n/a', 'nan', '-', '--', ''):
                return ''
            v = v.replace('&', 'and')
            if v == 'united states':
                return 'usa'
            if v == 'united kingdom':
                return 'uk'
            clean_num_str = v.replace(',', '').replace('=', '').replace('$', '').strip()
            try:
                return int(clean_num_str)
            except:
                pass
            try:
                return float(clean_num_str)
            except:
                pass
            return ' '.join(unidecode(v).split())
        return value

    def is_match(self, val_gt, val_pred):
        """Check if two values match: exact, numeric tolerance, or semantic similarity."""
        if val_gt == val_pred:
            return True

        if isinstance(val_gt, (int, float)) and isinstance(val_pred, (int, float)):
            if val_gt == 0:
                return abs(val_pred) < 1e-6
            return abs((val_pred - val_gt) / val_gt) <= 0.001

        if self.use_semantic and isinstance(val_gt, str) and isinstance(val_pred, str):
            if val_gt == '' or val_pred == '':
                return False
            emb1 = self.model.encode(val_gt, convert_to_tensor=True)
            emb2 = self.model.encode(val_pred, convert_to_tensor=True)
            return util.cos_sim(emb1, emb2).item() >= 0.5

        return False

    def evaluate(self, df_gt, df_pred, primary_columns, date_columns=None):
        if date_columns is None:
            date_columns = []

        # Normalize primary columns
        for col in primary_columns:
            is_date = col in date_columns
            df_gt[col] = df_gt[col].apply(lambda x: str(self.normalize_value(x, is_date)))
            df_pred[col] = df_pred[col].apply(lambda x: str(self.normalize_value(x, is_date)))

        # Deduplicate and set index for matching
        df_gt_indexed = df_gt.drop_duplicates(subset=primary_columns).set_index(primary_columns)
        df_pred_indexed = df_pred.drop_duplicates(subset=primary_columns).set_index(primary_columns)

        # Key matching
        common_indices = df_gt_indexed.index.intersection(df_pred_indexed.index)
        key_matches = len(common_indices)
        keys_recall = key_matches / len(df_gt_indexed) if len(df_gt_indexed) > 0 else 0
        keys_precision = key_matches / len(df_pred_indexed) if len(df_pred_indexed) > 0 else 0
        keys_f1 = 2 * keys_recall * keys_precision / (keys_recall + keys_precision) if (keys_recall + keys_precision) > 0 else 0

        # Non-key columns
        df_gt_aligned = df_gt_indexed.loc[common_indices]
        df_pred_aligned = df_pred_indexed.loc[common_indices]
        non_key_cols = [c for c in df_gt.columns if c not in primary_columns]

        correct_cells = 0
        for idx in common_indices:
            for col in non_key_cols:
                if col in df_pred_aligned.columns:
                    val_gt = self.normalize_value(df_gt_aligned.loc[idx, col])
                    val_pred = self.normalize_value(df_pred_aligned.loc[idx, col])
                    if self.is_match(val_gt, val_pred):
                        correct_cells += 1

        # Total metrics
        total_ops_gt = len(df_gt_indexed) * len(df_gt.columns)
        total_ops_pred = len(df_pred_indexed) * len(df_pred.columns)
        total_correct_ops = (key_matches * len(primary_columns)) + correct_cells

        tbl_recall = total_correct_ops / total_ops_gt if total_ops_gt > 0 else 0
        tbl_precision = total_correct_ops / total_ops_pred if total_ops_pred > 0 else 0
        tbl_f1 = 2 * tbl_recall * tbl_precision / (tbl_recall + tbl_precision) if (tbl_recall + tbl_precision) > 0 else 0

        # Non-key metrics
        total_non_key_gt = len(df_gt_indexed) * len(non_key_cols)
        total_non_key_pred = len(df_pred_indexed) * len(non_key_cols)
        nk_recall = correct_cells / total_non_key_gt if total_non_key_gt > 0 else 0
        nk_precision = correct_cells / total_non_key_pred if total_non_key_pred > 0 else 0
        nk_f1 = 2 * nk_recall * nk_precision / (nk_recall + nk_precision) if (nk_recall + nk_precision) > 0 else 0

        # Relative non-key accuracy
        rel_nk_acc = correct_cells / (key_matches * len(non_key_cols)) if key_matches > 0 else 0

        return {
            'keys_recall': keys_recall,
            'keys_precision': keys_precision,
            'keys_f1': keys_f1,
            'non_keys_recall': nk_recall,
            'non_keys_precision': nk_precision,
            'non_keys_f1': nk_f1,
            'table_recall': tbl_recall,
            'table_precision': tbl_precision,
            'table_f1': tbl_f1,
            'relative_non_key_accuracy': rel_nk_acc
        }