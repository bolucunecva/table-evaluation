# <img src="figures/t2t.png" width="80" alt="Text2TabEval"> Text2TabEval: A Python Library for Unified Evaluation of Text-to-Table Generation

[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://sisinflab.github.io/DataRec/)
[![License](https://img.shields.io/github/license/sisinflab/DataRec.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

---

<img src="figures/t2t.png" width="320" alt="Text2TabEval Logo">

**Text2TabEval** is an open-source Python library that provides a unified, multi-granularity evaluation framework for text-to-table (T2T) generation. It integrates string-based, numeric, embedding-based, and LLM-driven metrics with an analysis module for explainability.

## 📑 Table of Contents
- [Features](#features-)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Quickstart](#quickstart-)
- [Analysis Module](#analysis-module)
- [Datasets](#datasets)
- [Running Tests](#running-tests)
- [Documentation](#documentation-)
- [Contributing](#contributing-)
- [Authors](#authors-)


---

## Features ✨
- Unified evaluation framework for T2T generation models
- Metrics: EM, ChrF, BERTScore, ROUGE-L, Levenshtein, H-Score, CMT-Bench, P-Score, TabEval, TabXEval, RMSE
- Evaluation at cell, row, and table granularity
- Built-in benchmark datasets: E2E, Rotowire, WikiTableText + scientific domain sets
- Analysis module for metric explainability:
  - Metric agreement / divergence heatmaps
  - Error taxonomy (row shuffle, schema mismatch, numeric errors, etc.)
  - Per-table diagnosis with rule-based and LLM-based explanations
  - Context-aware metric recommendations

---

## Installation 🛠️ 
```bash
git clone https://github.com/bolucunecva/table-evaluation.git
cd table-evaluation

# Recommended: create a virtual environment first
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip

# Install the library
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```
---

## Environment Setup

If you use local HuggingFace models (required for LLM-based metrics):

```bash
# Set HuggingFace cache directory (optional but recommended)
export HF_HUB_CACHE=/path/to/your/model/cache

# Only needed for gated models (e.g. Llama, Gemma)
export HF_TOKEN=your_huggingface_token
```

---

## Quickstart 🚀
Below is a minimal example showing how to run evaluation:
```python
# Evaluate for evaluation of e2e dataset
# Importing required functions
from text2tabeval.datasets import load_gold_dataset, load_pred_dataset
from text2tabeval.non_llm import chrf_eval, bert_score_eval, cmt_eval, h_score_eval
from text2tabeval.llm import p_score_eval, tabeval, tabxeval

# --- Load dataset ---
dataset_name = "e2e"

# Ground truth dataset
gold_tables = load_gold_dataset(dataset_name)

# Model predictions
pred_tables = load_pred_dataset(f"text2tabeval/datasets/{dataset_name}/test.data")

# --- Non-LLM Evaluate ---
chrf_scores = chrf_eval.evaluate_tables(pred_tables, gold_tables)
bert_scores = bert_score_eval.evaluate_tables(pred_tables, gold_tables)
cmt_scores = cmt_eval.evaluate_tables(pred_tables, gold_tables)
h_scores = h_score_eval.evaluate_tables(pred_tables, gold_tables)

# --- LLM Evaluate ---
llm_local = LLMWrapper(
    backend="local",
    model_name=MODEL_PATH,
    device="cuda"
)

p_score_eval.evaluate_tables(pred_tables, gold_tables, llm_local)
tabeval_results = tabeval.evaluate_tables(
    pred_tables=pred_tables,
    gold_tables=gold_tables,
    llm=llm_local,
    nli_model="roberta-large-mnli", 
    device="cuda"
)
```
---

## Analysis Module

The analysis module explains *why* scores are what they are, across all four dimensions:

```python
import pandas as pd
from text2tabeval.analysis import AnalysisModule

# Build a scores DataFrame from your metric results
scores_df = pd.DataFrame({
    "ChrF":      [s["score"] for s in chrf_scores],
    "BERTScore": [s["score"] for s in bert_scores],
    "TabXEval":  [s["score"] for s in tabxeval_scores],
})

am = AnalysisModule(pred_tables, gold_tables, scores_df)

# 1. Metric agreement heatmap
am.metric_agreement.plot_heatmap(save_path="heatmap.png")

# 2. Error distribution
print(am.error_taxonomy.summary_report())

# 3. Per-table diagnosis
am.table_diagnosis.print_diagnosis(table_idx=5)

# 4. Metric recommendation for your setup
rec = am.metric_recommender.recommend(
    compute_budget="medium",
    domain="scientific",
    has_numeric_cols=True,
)
rec.pretty_print()

# Full combined report
print(am.full_report())
```

### Analysis Components

| Component | What it does |
|---|---|
| `MetricAgreement` | Spearman correlation matrix, metric clustering, divergence cases |
| `ErrorTaxonomy` | Per-table error classification, metric sensitivity per error type |
| `TableDiagnosis` | Rule-based + LLM explanation of why a table scored low |
| `MetricRecommender` | Context-aware metric selection grounded in TabXBench results |

---

## Datasets 📊

Built-in datasets:
| Dataset | Domain | Split |
|---|---|---|
| [E2E](https://aclanthology.org/W17-5525/) | Restaurant descriptions | test |
| [Rotowire](https://aclanthology.org/D17-1239/) | Sports (basketball) | test |
| [WikiTableText](https://aclanthology.org/D16-1128/) | Wikipedia tables | test |


Scientific domain datasets (also included):

| Dataset | Domain | Samples |
|---|---|---|
| MPEA | Materials Science | 65 |
| Diffusion | Materials Science | 51 |
| TDMS | Machine Learning | 330 |
| SciREX | Machine Learning | 371 |
   
---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Documentation 📚
Full documentation, examples, and API reference are available in the docs/ directory (or link to hosted docs if available).

---

## Contributing 🤝

Contributions are welcome!  
To contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-change`
3. Add tests in `tests/` for any new functionality.
4. Run `pytest tests/ -v` before pushing.
5. Open a pull request with a clear description.

---

## Authors 👥
- Necva Bolucu
- Stephen Wan

---








