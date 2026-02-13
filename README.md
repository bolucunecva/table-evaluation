# <img src="figures/t2t.png" width="80" alt="Text2TabEval"> Text2TabEval: A Python Library for Unified Evaluation of Text-to-Table Generation

[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://sisinflab.github.io/DataRec/)
[![License](https://img.shields.io/github/license/sisinflab/DataRec.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

---

<img src="figures/t2t.png" width="320" alt="Text2TabEval Logo">

**Text2TabEval** is an open-source library that unifies multiple evaluation perspectives to support reproducible and interpretable assessment of text-to-table generation systems.

## 📑 Table of Contents
- [Features](#features-)
- [Installation](#installation)
- [Quickstart](#quickstart-)
- [Usage Examples](#usage-examples)
- [Datasets](#datasets-)
- [Documentation](#documentation-)
- [Contributing](#contributing-)
- [Authors](#authors-)


---

## Features ✨
- Unified evaluation framework for text-to-table models
- Multiple automatic evaluators (semantic, structural, statistical)
- Simple Python API for loading predictions and gold tables
- Script and module-based usage for flexibility

---

## Installation 🛠️ 
Clone the repository and install dependencies:

```bash
git clone https://github.com/bolucunecva/table-evaluation.git
cd table-evaluation
pip install -r requirements.txt
```

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

## Datasets 📊

The datasets in the library: XXX, XXX

Scientific domain use case datasets:
- Machine Learning
  - TDMS
  - SciREX
- Materials Science
   - MPEA
   - Diffusion
   
---

## Documentation 📚
Full documentation, examples, and API reference are available in the docs/ directory (or link to hosted docs if available).

---

## Contributing 🤝

Contributions are welcome!  
To contribute:
1. Create a feature/fix branch.  
2. Add tests and documentation updates as needed.  
3. Run tests before pushing.  
4. Open a pull request describing your changes clearly.

---

## Authors 👥
- Necva Bolucu
- Stephen Wan

---







