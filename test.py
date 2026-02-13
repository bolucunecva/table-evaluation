from text2tabeval.datasets import load_gold_dataset, load_pred_dataset
from text2tabeval.non_llm import bert_score_eval, hierachical_eval, h_score_eval, cmt_eval, chrf_eval, exact_match_eval, general_metrics
from text2tabeval.llm import p_score_eval, tabeval, tabxeval
from text2tabeval.utils.llm_wrapper import LLMWrapper


## Scientific domain
gold_tables = dataset_loader.load_pred_dataset(f"scirex_gold.data")
pred_tables = dataset_loader.load_pred_dataset(f"scirex_pred.data")

# Non-LLM-based metrics
bert_score = bert_score_eval.evaluate_tables(pred_tables, gold_tables, row_header=False, col_header=False)

# --- Local HuggingFace model ---
llm_local = LLMWrapper(
    backend="local",
    model_name="../models/Qwen3-4B",
    device="cuda"
)

# LLM-based metrics
p_score = p_score_eval.evaluate_tables(pred_tables, gold_tables, llm_local)
tabeval_results = tabeval.evaluate_tables(
    pred_tables=pred_tables,
    gold_tables=gold_tables,
    llm=llm_local,    
    nli_model="roberta-large-mnli",
    device="cpu"
)


print("Scirex Aggregated Scores:", tabeval_results)
