# rmse_eval.py
import pandas as pd
import numpy as np

class RMSEEvaluator:
    """
    Evaluate tables using RMSE for numeric cells and Error Rate (ER%) for exact matching.
    """

    def __init__(self):
        pass

    @staticmethod
    def normalize_value(value):
        """Convert numeric-like strings to float, keep others as is."""
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                # Remove commas, dollar signs, equals
                v = value.replace(',', '').replace('$', '').replace('=', '').strip()
                return float(v)
            except:
                return np.nan
        return np.nan

    def evaluate(self, df_gt, df_pred):
        """
        Compute RMSE and Error Rate (ER%) for numeric tables.
        Flatten 2D tables to 1D sequences and compare element-wise.
        Non-numeric or missing cells are ignored for RMSE but counted for ER.
        """
        # Align shapes
        if df_gt.shape != df_pred.shape:
            raise ValueError(f"Ground truth and prediction must have same shape, got {df_gt.shape} vs {df_pred.shape}")

        # Flatten tables
        gt_values = df_gt.applymap(self.normalize_value).to_numpy().flatten()
        pred_values = df_pred.applymap(self.normalize_value).to_numpy().flatten()

        # RMSE: only for numeric cells
        mask_numeric = ~np.isnan(gt_values) & ~np.isnan(pred_values)
        n_numeric = np.sum(mask_numeric)
        if n_numeric > 0:
            mse = np.mean((gt_values[mask_numeric] - pred_values[mask_numeric])**2)
            rmse = np.sqrt(mse)
        else:
            rmse = np.nan

        # Error Rate (ER%): count exact mismatches including non-numeric
        total_cells = len(gt_values)
        errors = np.sum(gt_values != pred_values)
        er = (errors / total_cells) * 100 if total_cells > 0 else np.nan

        return {
            "RMSE": rmse,
            "Error_Rate_percent": er,
            "Total_cells": total_cells,
            "Numeric_cells": n_numeric,
            "Errors": errors
        }