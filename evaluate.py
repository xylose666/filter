import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_path(rel_path, cfg):
    path = Path(rel_path)
    if path.is_absolute():
        return path
    root = cfg.get("paths", {}).get("project_root") or cfg.get("project_root")
    candidates = []
    if root:
        candidates.append(Path(root) / path)
    candidates.extend([Path.cwd() / path, Path(__file__).parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def load_json_or_jsonl(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

def load_gold_answers(file_paths, cfg):
    gold = {}
    for fp in file_paths:
        path = resolve_path(fp, cfg)
        if not path.exists():
            continue
        loaded = load_json_or_jsonl(path)
        if loaded is None:
            continue
        if isinstance(loaded, dict):
            loaded = [loaded]
        for obj in loaded:
            if not isinstance(obj, dict):
                continue
            qid = obj.get("id")
            ans = obj.get("answer")
            if qid is None or ans is None:
                continue
            gold.setdefault(qid, [])
            if isinstance(ans, list):
                for item in ans:
                    normalized = normalize_text(item)
                    if normalized:
                        gold[qid].append(normalized)
            else:
                normalized = normalize_text(ans)
                if normalized:
                    gold[qid].append(normalized)
    for qid in gold:
        gold[qid] = list(set(gold[qid]))
    return gold

def normalize_text(text):
    if isinstance(text, list):
        return [normalize_text(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
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
    gold_set = set(normalize_text(g) for g in gold_list if g)
    if gen_norm in gold_set:
        return True
    for g in gold_list:
        gold_norm = normalize_text(g)
        if not gold_norm:
            continue
        gen_tokens = set(gen_norm.split())
        gold_tokens = set(gold_norm.split())
        if gen_tokens and gold_tokens and (gen_tokens <= gold_tokens or gold_tokens <= gen_tokens):
            return True
        if gold_norm in gen_norm or gen_norm in gold_norm:
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
    filtered_results_path = resolve_path(cfg["output"]["filtered_results"], cfg)
    if not filtered_results_path.exists():
        print(f"Filtered results not found at {filtered_results_path}. Run filter.py first.")
        return
    with open(filtered_results_path, "r", encoding="utf-8") as f:
        filtered = json.load(f)
    gold = load_gold_answers(paths["gold_answers"], cfg)
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
    out_path = resolve_path(cfg["output"]["evaluation"], cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
