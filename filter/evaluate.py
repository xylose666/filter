import json
import re
from pathlib import Path
from collections import defaultdict

CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_gold_answers(file_paths):
    gold = {}
    for fp in file_paths:
        if not Path(fp).exists():
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    qid = obj.get("id")
                    ans = obj.get("answer")
                    if qid is not None and ans is not None:
                        if qid not in gold:
                            gold[qid] = []
                        if isinstance(ans, list):
                            gold[qid].extend([str(a).lower().strip() for a in ans])
                        else:
                            gold[qid].append(str(ans).lower().strip())
                except:
                    continue
    for qid in gold:
        gold[qid] = list(set(gold[qid]))
    return gold

def normalize_text(text):
    if isinstance(text, list):
        return [normalize_text(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def match_answer(generated, gold_list):
    if not generated:
        return False
    gen_norm = normalize_text(generated)
    if not gen_norm:
        return False
    gold_set = set(gold_list)
    if gen_norm in gold_set:
        return True
    for g in gold_list:
        if g in gen_norm or gen_norm in g:
            return True
    return False

def compute_metrics(predictions):
    total = len(predictions)
    if total == 0:
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}
    correct = sum(1 for p in predictions if p["match"])
    accuracy = correct / total
    precision = correct / total
    recall = correct / total
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
        "correct": correct
    }

def main():
    cfg = load_config()
    paths = cfg["paths"]
    filtered_results_path = Path(cfg["output"]["filtered_results"])
    if not filtered_results_path.exists():
        print(f"Filtered results not found at {filtered_results_path}. Run filter.py first.")
        return
    with open(filtered_results_path, "r", encoding="utf-8") as f:
        filtered = json.load(f)
    gold = load_gold_answers(paths["gold_answers"])
    eval_results = []
    for item in filtered:
        qid = item["question_id"]
        final_ans = item.get("final_answer", "")
        if isinstance(final_ans, list):
            final_str = " ".join(str(x) for x in final_ans)
        else:
            final_str = str(final_ans)
        gold_ans = gold.get(qid, [])
        match = match_answer(final_str, gold_ans) if gold_ans else False
        eval_results.append({
            "question_id": qid,
            "gold_answer": gold_ans,
            "predicted_answer": final_str,
            "match": match
        })
    metrics = compute_metrics(eval_results)
    output = {
        "metrics": metrics,
        "details": eval_results
    }
    out_path = Path(cfg["output"]["evaluation"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()