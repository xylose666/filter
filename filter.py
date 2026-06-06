import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import dashscope
import jieba.posseg as pseg
from dashscope import Generation

CONFIG_PATH = Path(__file__).parent / "config.json"

DEFAULT_SPORT_DICT = {
    "equestrian",
    "dressage",
    "jumping",
    "eventing",
    "shooting",
    "badminton",
    "fencing",
    "basketball",
    "diving",
    "swimming",
    "athletics",
    "pentathlon",
    "baseball",
    "hockey",
    "gymnastics",
    "rowing",
    "sailing",
    "weightlifting",
    "wrestling",
    "judo",
    "boxing",
    "archery",
    "triathlon",
    "volleyball",
    "tennis",
    "football",
    "soccer",
    "cycling",
    "skating",
    "skiing",
    "biathlon",
    "curling",
    "bobsleigh",
    "luge",
    "trampoline",
    "马术",
    "盛装舞步",
    "跳跃",
    "三项赛",
    "射击",
    "羽毛球",
    "击剑",
    "篮球",
    "跳水",
    "游泳",
    "田径",
    "现代五项",
    "棒球",
    "曲棍球",
    "体操",
    "赛艇",
    "帆船",
    "举重",
    "摔跤",
    "柔道",
    "拳击",
    "射箭",
    "铁人三项",
    "排球",
    "网球",
    "足球",
    "自行车",
    "滑冰",
    "滑雪",
    "冬季两项",
    "冰壶",
    "雪车",
    "雪橇",
    "蹦床",
}

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "by",
    "at",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "but",
    "or",
    "so",
    "yet",
    "nor",
    "also",
    "very",
    "just",
    "can",
    "will",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "this",
    "that",
    "these",
    "those",
    "then",
    "than",
    "such",
    "both",
    "each",
    "some",
    "any",
    "no",
    "many",
    "more",
    "most",
    "few",
    "fewer",
    "less",
    "enough",
    "all",
    "every",
    "several",
    "either",
    "neither",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "up",
    "down",
    "off",
    "over",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "where",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "using",
    "including",
    "only",
    "own",
    "same",
    "without",
    "would",
    "could",
}

META_KEY_EXACT = {
    "id",
    "result_id",
    "created_at",
    "updated_at",
    "input",
    "output",
    "results",
    "retrieval_results",
    "used_json",
    "used_tables",
    "generated_sql",
    "refined_sql",
    "question",
    "sql",
    "k",
}

SPLIT_PATTERN = re.compile(r"[、,，;；|/\n]+")
WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z'\-]*")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(rel_path, cfg):
    p = Path(rel_path)
    if p.is_absolute():
        return p

    candidates = []
    root = cfg.get("paths", {}).get("project_root") or cfg.get("project_root")
    if root:
        candidates.append(Path(root) / p)
    candidates.extend(
        [
            Path.cwd() / p,
            Path(__file__).parent / p,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def normalize_scalar(value):
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\u00a0", " ")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_match(text):
    text = normalize_scalar(text).lower()
    text = re.sub(r"[^\w\s\u4e00-\u9fff-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item is None:
            continue
        cleaned = normalize_scalar(item)
        if not cleaned:
            continue
        key = normalize_for_match(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def should_skip_key(key):
    lower = str(key).lower()
    if lower in META_KEY_EXACT:
        return True
    if lower.endswith("_id"):
        return True
    if "sql" in lower:
        return True
    return False


def split_scalar_text(text):
    parts = [piece.strip() for piece in SPLIT_PATTERN.split(text) if piece.strip()]
    if len(parts) > 1:
        return parts
    return [text.strip()]


def extract_answer_candidates(value, depth=0):
    if value is None:
        return []

    if isinstance(value, dict):
        candidates = []

        if "answer" in value:
            candidates.extend(extract_answer_candidates(value.get("answer"), depth + 1))
        if "text" in value and isinstance(value.get("text"), str):
            candidates.extend(extract_answer_candidates(value.get("text"), depth + 1))
        if "content" in value:
            candidates.extend(extract_answer_candidates(value.get("content"), depth + 1))
        if "data" in value and isinstance(value.get("data"), dict):
            data = value.get("data") or {}
            for key, child in data.items():
                if should_skip_key(key):
                    continue
                candidates.extend(extract_answer_candidates(child, depth + 1))

        if candidates:
            return candidates

        for key, child in value.items():
            if should_skip_key(key):
                continue
            candidates.extend(extract_answer_candidates(child, depth + 1))
        return candidates

    if isinstance(value, (list, tuple, set)):
        candidates = []
        for item in value:
            candidates.extend(extract_answer_candidates(item, depth + 1))
        return candidates

    text = normalize_scalar(value)
    if not text:
        return []
    if len(text) > 1:
        return split_scalar_text(text)
    return [text]


def load_json_or_jsonl(file_path):
    path = Path(file_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        result = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except Exception:
                continue
        return result


def build_sport_dict(cfg):
    sport_dict_path = resolve_path(cfg["output"]["sport_dict_cache"], cfg)
    if sport_dict_path.exists():
        try:
            with open(sport_dict_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data:
                return set(data)
        except Exception:
            pass

    all_questions = []
    for qfile in cfg["paths"]["gold_answers"]:
        path = resolve_path(qfile, cfg)
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
            question = obj.get("question", "")
            if question:
                all_questions.append(question)

    if not all_questions:
        return DEFAULT_SPORT_DICT.copy()

    word_counter = Counter()
    for question in all_questions:
        if re.search(r"[\u4e00-\u9fff]", question):
            for word, flag in pseg.cut(question):
                if len(word) > 1 and flag.startswith("n"):
                    word_counter[word] += 1
        else:
            for word in WORD_PATTERN.findall(normalize_for_match(question)):
                if len(word) > 2 and word not in DEFAULT_STOPWORDS:
                    word_counter[word] += 1

    stop_words = {
        "olympics",
        "olympic",
        "medal",
        "medals",
        "event",
        "events",
        "athlete",
        "athletes",
        "player",
        "players",
        "team",
        "teams",
        "country",
        "countries",
        "games",
        "title",
        "titles",
        "gold",
        "silver",
        "bronze",
        "competition",
        "competitions",
        "winner",
        "winners",
        "won",
        "list",
        "lists",
    }

    sport_dict = set()
    for word, count in word_counter.most_common(300):
        if count >= 2 and word not in stop_words:
            sport_dict.add(word)

    if len(sport_dict) < 10:
        sport_dict.update(DEFAULT_SPORT_DICT)

    sport_dict_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sport_dict_path, "w", encoding="utf-8") as handle:
        json.dump(sorted(sport_dict), handle, ensure_ascii=False, indent=2)

    return sport_dict


def load_refiner_json(file_path, cfg):
    path = resolve_path(file_path, cfg)
    if not path.exists():
        return {}

    loaded = load_json_or_jsonl(path)
    if loaded is None:
        return {}
    if isinstance(loaded, dict):
        loaded = [loaded]

    result_by_id = {}
    for item in loaded:
        if not isinstance(item, dict):
            continue
        used = item.get("used_json") or item.get("input") or {}
        qid = used.get("id") if isinstance(used, dict) else item.get("question_id")
        if qid is None:
            continue

        answers = []
        for result in item.get("results", []) or []:
            answers.extend(extract_answer_candidates(result.get("answer")))
            if not answers and isinstance(result, dict):
                answers.extend(extract_answer_candidates(result.get("data")))
                answers.extend(extract_answer_candidates(result.get("text")))

        if not answers:
            answers.extend(extract_answer_candidates(item.get("final_results")))

        result_by_id[qid] = {
            "final_results": dedupe_preserve_order(answers)[:20],
            "refined_sql": item.get("refined_sql", ""),
            "iterations": item.get("iterations", 1),
        }

    return result_by_id


def load_jsonl(file_path, cfg, id_key="id"):
    path = resolve_path(file_path, cfg)
    data = {}
    if not path.exists():
        return data

    loaded = load_json_or_jsonl(path)
    if loaded is None:
        return data
    if isinstance(loaded, dict):
        loaded = [loaded]

    if isinstance(loaded, list):
        for index, obj in enumerate(loaded):
            if not isinstance(obj, dict):
                continue
            if id_key is None:
                data[index] = obj
                continue
            obj_id = obj.get(id_key)
            if obj_id is not None:
                data[obj_id] = obj
        return data

    return data


def extract_keywords(text, sport_dict):
    words = set()
    if not text:
        return words, []

    normalized = normalize_scalar(text)
    if re.search(r"[\u4e00-\u9fff]", normalized):
        for word, flag in pseg.cut(normalized):
            if len(word) > 1 and (flag.startswith("n") or flag in {"eng", "nr", "ns", "nt", "nz"}):
                words.add(word)
        for word in WORD_PATTERN.findall(normalize_for_match(normalized)):
            if len(word) > 2 and word not in DEFAULT_STOPWORDS:
                words.add(word)
    else:
        for word in WORD_PATTERN.findall(normalize_for_match(normalized)):
            if len(word) > 2 and word not in DEFAULT_STOPWORDS:
                words.add(word)

    sport_flags = [word for word in words if word in sport_dict]
    return words, sport_flags


def score_semantic_relevance(sub_question, text, sport_dict, sport_bonus_weight=0.15):
    if not text:
        return 0.0
    sq_words, sq_sports = extract_keywords(sub_question, sport_dict)
    text_words, text_sports = extract_keywords(text, sport_dict)
    if not sq_words:
        return 0.3

    overlap = len(sq_words & text_words) / max(1, len(sq_words))
    if overlap == 0.0:
        question_norm = normalize_for_match(sub_question)
        text_norm = normalize_for_match(text)
        for token in list(sq_words)[:5]:
            token_norm = normalize_for_match(token)
            if token_norm and (token_norm in text_norm or text_norm in token_norm):
                overlap = 0.15
                break

    sport_bonus = sport_bonus_weight if (sq_sports and text_sports) else 0.0
    return min(1.0, overlap + sport_bonus)


def collect_sql_candidates(records):
    candidates = []
    for record in records or []:
        if not isinstance(record, dict):
            candidates.extend(extract_answer_candidates(record))
            continue

        output = record.get("output") or {}
        candidates.extend(extract_answer_candidates(output.get("results")))
        candidates.extend(extract_answer_candidates(output.get("answer")))
        candidates.extend(extract_answer_candidates(output.get("data")))
        candidates.extend(extract_answer_candidates(record.get("results")))
        candidates.extend(extract_answer_candidates(record.get("answer")))
        candidates.extend(extract_answer_candidates(record.get("data")))

    return dedupe_preserve_order(candidates)


def filter_sql_results(sub_question, question_id, sql_by_id, refiner_map, cfg, sport_dict):
    refiner = refiner_map.get(question_id, {})
    refiner_results = refiner.get("final_results") or []
    if refiner_results:
        return dedupe_preserve_order(refiner_results)[:20], 1.0

    records = sql_by_id.get(question_id, [])
    candidates = collect_sql_candidates(records)
    if not candidates:
        return [], 0.0

    scored = []
    for index, candidate in enumerate(candidates):
        relevance = score_semantic_relevance(sub_question, candidate, sport_dict)
        scored.append(
            {
                "text": candidate,
                "score": relevance + max(0.0, 0.15 - index * 0.01),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    best = [item["text"] for item in scored[:20]]
    best_score = scored[0]["score"] if scored else 0.0
    return dedupe_preserve_order(best), best_score


def get_vector_results(sub_question, question_id, vector_map, cfg, sport_dict):
    vec = vector_map.get(question_id, {})
    retrieval = vec.get("retrieval_results", {}) or {}
    thresholds = cfg.get("thresholds", {})

    top_k = max(5, int(thresholds.get("vector_top_k_per_type", 5)))
    min_rel = float(thresholds.get("vector_relevance_min", 0.1))
    sport_bonus = float(thresholds.get("sport_bonus_weight", 0.15))

    type_weight = {
        "text": 1.0,
        "infobox": 1.15,
        "table": 1.1,
    }

    candidates = []
    for result_type in ["text", "infobox", "table"]:
        bucket = retrieval.get(result_type, {}) or {}
        items = bucket.get("results", []) or []
        if not items:
            continue

        raw_scores = [float(item.get("score", 0) or 0) for item in items]
        max_raw = max(raw_scores) if raw_scores else 0.0

        typed_candidates = []
        for item in items:
            raw_score = float(item.get("score", 0) or 0)
            content = item.get("content", {}) or {}
            if result_type == "text":
                text = content.get("text", "") if isinstance(content, dict) else str(content)
            else:
                if isinstance(content, dict):
                    text = " | ".join(f"{key}: {value}" for key, value in content.items() if value is not None)
                else:
                    text = str(content)

            text = normalize_scalar(text)
            if not text:
                continue

            relevance = score_semantic_relevance(sub_question, text, sport_dict, sport_bonus)
            normalized_raw = raw_score / max_raw if max_raw > 0 else 0.0
            score = relevance * 0.6 + normalized_raw * 0.3 + type_weight.get(result_type, 1.0) * 0.05
            if raw_score >= float(thresholds.get(f"vector_min_score_{result_type}", 0) or 0):
                score += 0.05
            typed_candidates.append(
                {
                    "text": text,
                    "relevance": relevance,
                    "score": score,
                    "type": result_type,
                }
            )

        typed_candidates.sort(key=lambda item: item["score"], reverse=True)
        strong_candidates = [item for item in typed_candidates if item["relevance"] >= min_rel]
        candidates.extend(strong_candidates[:top_k] if strong_candidates else typed_candidates[:top_k])

    if not candidates:
        return []

    candidates.sort(key=lambda item: item["score"], reverse=True)
    texts = [item["text"] for item in candidates[: max(top_k, 10)]]
    return dedupe_preserve_order(texts)


def call_llm(prompt, model, api_key, temperature=0):
    dashscope.api_key = api_key
    try:
        response = Generation.call(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            result_format="message",
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content.strip()
        print(f"LLM API error: {response.message}")
        return ""
    except Exception as exc:
        print(f"LLM call failed: {exc}")
        return ""


def format_answer_list(values, limit=10):
    values = dedupe_preserve_order(values)[:limit]
    return "、".join(values)


def answer_sub_question_with_llm(sub_q, filtered_answers, llm_config, prompts):
    if not filtered_answers:
        return "Information not found."
    prompt = prompts["answer_sub_question"].format(
        question=sub_q,
        retrieved_info=json.dumps(filtered_answers[:10], ensure_ascii=False),
    )
    return call_llm(prompt, llm_config["model"], llm_config["api_key"], llm_config["temperature"])


def fuse_answers_with_llm(original_question, sub_answers, combination, llm_config, prompts):
    combo_type = combination.get("type", "parallel")
    combo_desc = combination.get("description", "")
    prompt = prompts["fuse_answers"].format(
        original_question=original_question,
        combination_type=combo_type,
        combination_description=combo_desc,
        sub_answers=json.dumps(sub_answers, ensure_ascii=False, indent=2),
    )
    return call_llm(prompt, llm_config["model"], llm_config["api_key"], llm_config["temperature"])


def process_question(question_id, data_maps, cfg, sport_dict):
    decomp = data_maps["decomposer"].get(question_id, {})
    original_q = decomp.get("original_question", "")
    sub_questions = decomp.get("sub_questions", [])
    combination = decomp.get("combination", {}) or {}
    if not sub_questions:
        sub_questions = [original_q]

    llm_config = cfg.get("llm", {})
    prompts = cfg.get("prompts", {})
    has_llm = bool(llm_config.get("api_key") and prompts)

    sub_answers = []
    for sub_question in sub_questions:
        sql_ans, _ = filter_sql_results(
            sub_question,
            question_id,
            data_maps["sql_exec"],
            data_maps["refiner"],
            cfg,
            sport_dict,
        )
        vec_ans = get_vector_results(sub_question, question_id, data_maps["vector"], cfg, sport_dict)
        merged = dedupe_preserve_order(sql_ans + vec_ans)

        if has_llm and merged:
            answer = answer_sub_question_with_llm(sub_question, merged, llm_config, prompts)
        elif merged:
            answer = format_answer_list(merged, limit=10)
        else:
            answer = ""
        sub_answers.append(answer)

    if has_llm and len(sub_answers) > 1:
        final_answer = fuse_answers_with_llm(original_q, sub_answers, combination, llm_config, prompts)
    elif len(sub_answers) > 1:
        final_answer = "；".join(answer for answer in sub_answers if answer)
    else:
        final_answer = sub_answers[0] if sub_answers else ""

    return {
        "question_id": question_id,
        "original_question": original_q,
        "sub_questions": sub_questions,
        "combination_type": combination.get("type", "parallel"),
        "sub_answers": sub_answers,
        "final_answer": final_answer,
    }


def parse_limit(argv):
    limit = 10
    if len(argv) <= 1:
        return limit

    if argv[1] == "--mode" and len(argv) > 2:
        mode = argv[2].lower()
        if mode == "full":
            return None
        if mode == "test":
            return 10

    if argv[1].startswith("--limit="):
        try:
            return int(argv[1].split("=", 1)[1])
        except ValueError:
            return 10

    return 10


def main():
    cfg = load_config()
    paths = cfg["paths"]

    limit = parse_limit(sys.argv)
    print(f"Mode: {'full' if limit is None else f'test({limit})'}")

    sport_dict = build_sport_dict(cfg)
    print(f"Sport dictionary size: {len(sport_dict)}")

    decomp_map = load_jsonl(paths["decomposer"], cfg)
    vector_map = load_jsonl(paths["vector_retrieval"], cfg)
    sql_records = load_jsonl(paths["sql_execution"], cfg, id_key=None)

    sql_by_id = {}
    for record in sql_records.values():
        if not isinstance(record, dict):
            continue
        inp = record.get("input") or {}
        qid = inp.get("id")
        if qid is not None:
            sql_by_id.setdefault(qid, []).append(record)

    refiner_map = load_refiner_json(paths["refiner"], cfg)

    data_maps = {
        "decomposer": decomp_map,
        "sql_exec": sql_by_id,
        "refiner": refiner_map,
        "vector": vector_map,
    }

    all_qids = sorted(set(decomp_map.keys()) | set(vector_map.keys()) | set(sql_by_id.keys()))
    if limit is not None:
        all_qids = all_qids[:limit]

    results = []
    for qid in all_qids:
        results.append(process_question(qid, data_maps, cfg, sport_dict))

    out_path = resolve_path(cfg["output"]["filtered_results"], cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(f"Filtered {len(results)} questions. Saved to {out_path}")


if __name__ == "__main__":
    main()
