import json
import re
import numpy as np
from pathlib import Path

# ------------------
# P-Score (LLM)
# ------------------
def _score_once(llm, prompt_template, t1, t2):
    prompt = prompt_template.format(table1=t1, table2=t2)
    out = llm.generate(prompt)

    scores = {}
    try:
        # Try direct JSON parse
        scores = json.loads(out)
    except json.JSONDecodeError:
        # Extract the first {...} JSON-like object in output
        m = re.search(r"\{.*\}", out, re.S)
        if m:
            try:
                scores = json.loads(m.group())
            except json.JSONDecodeError:
                # fallback if still invalid
                scores = {}

    c = float(scores.get("content_similarity", 0))
    s = float(scores.get("structural_similarity", 0))
    # Clip to [0,10]
    return max(0, min(10, c)), max(0, min(10, s))

def evaluate_tables(
    pred_tables,
    gold_tables,
    llm,
    prompt_path=None,
    bidirectional=True
):
    if prompt_path is None:
        prompt_path = Path(__file__).parent.parent / "prompts" / "pscore.txt"

    prompt_template = Path(prompt_path).read_text(encoding="utf-8")

    contents, structures = [], []

    for g, p in zip(gold_tables, pred_tables):
        c1, s1 = _score_once(llm, prompt_template, g, p)

        if bidirectional:
            c2, s2 = _score_once(llm, prompt_template, p, g)
            c, s = (c1 + c2) / 2, (s1 + s2) / 2
        else:
            c, s = c1, s1

        contents.append(c)
        structures.append(s)

    return {
        "p_content_similarity": round(float(np.mean(contents)), 2),
        "p_structural_similarity": round(float(np.mean(structures)), 2),
    }
