import gradio as gr
import pandas as pd
# Assuming these imports work in your local environment
from text_to_table_evaluation import evaluate_tables
from semantic_eval import TableEvaluator
from rmse_eval import RMSEEvaluator
from hierarchical_table_evaluator import HierarchicalTableEvaluator

# Step 1: Load CSVs and extract column headers
def load_csvs(gt_file, pred_file):
    # Check if files are uploaded
    if gt_file is None or pred_file is None:
        return gr.CheckboxGroup(choices=[]), None, None
    
    # FIX 1: Read directly from the file path string (removed .name)
    try:
        df_gt = pd.read_csv(gt_file)
        df_pred = pd.read_csv(pred_file)
    except Exception as e:
        # Fallback in case of file read errors
        print(f"Error reading CSV: {e}")
        return gr.CheckboxGroup(choices=[]), None, None

    columns = list(df_gt.columns)
    
    # FIX 2: Return the component class directly instead of gr.update
    return gr.CheckboxGroup(choices=columns), df_gt, df_pred 

# Step 2: Run evaluation
def run_eval(eval_method, metric_choice, numeric_cols, df_gt, df_pred):
    if df_gt is None or df_pred is None:
        return "Please upload CSV files first."
    
    if eval_method == "Standard Table Eval":
        results = evaluate_tables(
            hyp_df=df_pred,
            tgt_df=df_gt,
            row_header=True,
            col_header=True,
            metric=metric_choice
        )
        
        return "\n".join([f"{k}\t: {v:.4f}" for k, v in results.items()])                
    elif eval_method == "Semantic Eval":
        evaluator = TableEvaluator()
        results = evaluator.evaluate(df_gt, df_pred, primary_columns=list(df_gt.columns))
        return "\n".join([f"{k}: {v:.4f}" for k, v in results.items()])
    elif eval_method == "RMSE + Error Rate":
        if not numeric_cols:
            return "Please select numeric columns for RMSE."
        # Ensure we only select columns that actually exist (intersection)
        valid_cols = [c for c in numeric_cols if c in df_gt.columns]
        
        df_gt_num = df_gt[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df_pred_num = df_pred[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        evaluator = RMSEEvaluator()
        results = evaluator.evaluate(df_gt_num, df_pred_num)
        return "\n".join([f"{k}: {v:.4f}" for k, v in results.items()])
    elif eval_method == "HierarchicalTableEvaluator":
        evaluator = HierarchicalTableEvaluator()
        results = evaluator.evaluate(df_gt, df_pred)
        return "\n".join([f"{k}: {v:.4f}" for k, v in results.items()])       

with gr.Blocks() as demo:
    gr.Markdown("## Unified Table Evaluation")
    
    with gr.Row():
        # type="filepath" is the default in Gradio 4, but good to be explicit
        gt_file = gr.File(label="Ground Truth CSV", type="filepath") 
        pred_file = gr.File(label="Predicted CSV", type="filepath")
    
    load_button = gr.Button("Load CSVs")
    
    # Initialize empty CheckboxGroup
    numeric_cols = gr.CheckboxGroup(label="Select Numeric Columns for RMSE", choices=[])
    
    # Hidden states to store DataFrames
    df_gt_state = gr.State()
    df_pred_state = gr.State()
    
    # Populate numeric column checkboxes
    load_button.click(
        fn=load_csvs,
        inputs=[gt_file, pred_file],
        outputs=[numeric_cols, df_gt_state, df_pred_state]
    )
    
    eval_method = gr.Radio(
        ["Standard Table Eval", "Semantic Eval", "RMSE + Error Rate", "HierarchicalTableEvaluator"],
        label="Choose Evaluation Method"
    )
    
    metric_choice = gr.Dropdown(
        ["E", "c", "BS-scaled"], 
        label="Metric for Standard Eval", 
        value="E"
    )
    
    run_button = gr.Button("Run Evaluation")
    result_box = gr.Textbox(label="Evaluation Results", lines=15)
    
    run_button.click(
        fn=run_eval,
        inputs=[eval_method, metric_choice, numeric_cols, df_gt_state, df_pred_state],
        outputs=[result_box]
    )

demo.launch()