# filter/filter.py
import json
import re
import sys
from pathlib import Path
from collections import Counter
import jieba
import jieba.posseg as pseg
import dashscope
from dashscope import Generation

CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_path(rel_path, cfg):
    """解析路径，优先使用 project_root 拼接，若不存在则尝试作为绝对路径或相对当前工作目录"""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    root = cfg.get("project_root")
    if root:
        candidate = Path(root) / p
        if candidate.exists():
            return candidate
    # 回退到当前工作目录或相对于 filter.py 的父目录
    candidates = [
        Path.cwd() / p,
        Path(__file__).parent.parent / p
    ]
    for c in candidates:
        if c.exists():
            return c
    # 如果都不存在，返回第一个候选（文件可能稍后创建）
    return candidates[0]

# ---------- 内置默认运动词典 ----------
DEFAULT_SPORT_DICT = set([
    'equestrian', 'dressage', 'jumping', 'eventing', 'shooting', 'badminton', 'fencing',
    'basketball', 'diving', 'swimming', 'athletics', 'pentathlon', 'baseball', 'hockey',
    'gymnastics', 'rowing', 'sailing', 'weightlifting', 'wrestling', 'judo', 'boxing',
    'archery', 'triathlon', 'volleyball', 'tennis', 'football', 'soccer', 'cycling',
    'skating', 'skiing', 'biathlon', 'curling', 'bobsleigh', 'luge', 'trampoline',
    '马术', '盛装舞步', '跳跃', '三项赛', '射击', '羽毛球', '击剑', '篮球', '跳水',
    '游泳', '田径', '现代五项', '棒球', '曲棍球', '体操', '赛艇', '帆船', '举重',
    '摔跤', '柔道', '拳击', '射箭', '铁人三项', '排球', '网球', '足球', '自行车',
    '滑冰', '滑雪', '冬季两项', '冰壶', '雪车', '雪橇', '蹦床'
])

# ---------- 运动词典自动构建 ----------
def build_sport_dict(cfg):
    sport_dict_path = resolve_path(cfg["output"]["sport_dict_cache"], cfg)
    if sport_dict_path.exists():
        try:
            with open(sport_dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    return set(data)
        except:
            pass  # 缓存损坏，重新构建

    all_questions = []
    for qfile in cfg["paths"]["gold_answers"]:
        p = resolve_path(qfile, cfg)
        if not p.exists():
            print(f"警告: 问题文件不存在 {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q = obj.get("question", "")
                    if q:
                        all_questions.append(q)
                except:
                    continue
    print(f"从 {len(cfg['paths']['gold_answers'])} 个文件加载问题数量: {len(all_questions)}")

    # 如果没加载到任何问题，直接返回默认词典
    if not all_questions:
        print("未加载到任何问题，使用默认体育词典")
        return DEFAULT_SPORT_DICT.copy()

    # 提取关键词
    word_counter = Counter()
    for q in all_questions:
        # 检测是否包含中文
        if re.search(r'[\u4e00-\u9fff]', q):
            words = pseg.cut(q)
            for w, flag in words:
                if flag.startswith('n') and len(w) > 1:
                    word_counter[w] += 1
        else:
            # 英文问题：基于空格分词，过滤停用词
            stopwords = set(['a', 'an', 'the', 'and', 'of', 'to', 'in', 'for', 'on', 'with', 'by', 'at', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'but', 'or', 'so', 'for', 'yet', 'nor', 'also', 'very', 'just', 'can', 'will', 'shall', 'should', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'then', 'than', 'such', 'both', 'each', 'some', 'any', 'no', 'many', 'more', 'most', 'few', 'fewer', 'less', 'enough', 'all', 'every', 'several', 'both', 'either', 'neither', 'what', 'which', 'who', 'whom', 'whose', 'why', 'how', 'up', 'down', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'where', 'when', 'where', 'why', 'how', 'about', 'than', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'using', 'also', 'including', 'such', 'only', 'own', 'same', 'so', 'than', 'that', 'then', 'these', 'those', 'through', 'very', 'without', 'will', 'with', 'would', 'could', 'should', 'might'])
            for w in q.lower().split():
                w = w.strip('.,!?;:()[]{}"\'')
                if len(w) > 2 and w not in stopwords and w.isalpha():
                    word_counter[w] += 1

    # 过滤常见非体育词
    stop_words = {"奥运会", "奥林匹克", "奥运", "项目", "比赛", "运动员", "金牌", "银牌", "铜牌", "奖牌",
                  "成绩", "决赛", "预赛", "资格", "冠军", "亚军", "季军", "选手", "队伍", "国家", "地区",
                  "时间", "地点", "年份", "历史", "规则", "记录", "世界", "中国", "美国", "英国", "法国"}
    sport_dict = set()
    for word, count in word_counter.most_common(200):
        if word not in stop_words and count >= 2:
            sport_dict.add(word)

    # 如果提取结果太少，补充默认词典
    if len(sport_dict) < 10:
        print(f"提取的体育词太少 ({len(sport_dict)})，补充默认词典")
        sport_dict.update(DEFAULT_SPORT_DICT)

    # 保存缓存
    sport_dict_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sport_dict_path, "w", encoding="utf-8") as f:
        json.dump(list(sport_dict), f, ensure_ascii=False)
    print(f"最终运动词典大小: {len(sport_dict)}")
    return sport_dict

# ---------- 加载 Refiner JSON ----------
def load_refiner_json(file_path, cfg):
    p = resolve_path(file_path, cfg)
    if not p.exists():
        return {}
    result_by_id = {}
    with open(p, "r", encoding="utf-8") as f:
        # 尝试整体解析为数组
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    used = item.get("used_json", {})
                    qid = used.get("id")
                    if qid is not None:
                        answers = []
                        for res in item.get("results", []):
                            ans = res.get("answer")
                            if ans and isinstance(ans, list):
                                answers.extend(ans)
                        result_by_id[qid] = {
                            "final_results": answers,
                            "refined_sql": item.get("refined_sql", ""),
                            "iterations": 1
                        }
        except json.JSONDecodeError:
            # 按行解析（JSONL）
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    used = item.get("used_json", {})
                    qid = used.get("id")
                    if qid is not None:
                        answers = []
                        for res in item.get("results", []):
                            ans = res.get("answer")
                            if ans and isinstance(ans, list):
                                answers.extend(ans)
                        result_by_id[qid] = {
                            "final_results": answers,
                            "refined_sql": item.get("refined_sql", ""),
                            "iterations": 1
                        }
                except:
                    continue
    return result_by_id

# ---------- 工具函数 ----------
def load_jsonl(file_path, cfg, id_key="id"):
    p = resolve_path(file_path, cfg)
    data = {}
    if not p.exists():
        return data
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj_id = obj.get(id_key)
                if obj_id is not None:
                    data[obj_id] = obj
            except:
                continue
    return data

def extract_keywords(text, sport_dict):
    """提取文本中的关键词，支持中英文"""
    words = set()
    # 检测是否包含中文
    if re.search(r'[\u4e00-\u9fff]', text):
        seg = pseg.cut(text)
        for w, flag in seg:
            if flag.startswith('n') and len(w) > 1:
                words.add(w)
    else:
        stopwords = {'a', 'an', 'the', 'and', 'of', 'to', 'in', 'for', 'on', 'with', 'by', 'at', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'but', 'or', 'so', 'for', 'yet', 'nor', 'also', 'very', 'just', 'can', 'will', 'shall', 'should', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'then', 'than', 'such', 'both', 'each', 'some', 'any', 'no', 'many', 'more', 'most', 'few', 'fewer', 'less', 'enough', 'all', 'every', 'several', 'both', 'either', 'neither', 'what', 'which', 'who', 'whom', 'whose', 'why', 'how', 'up', 'down', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'where', 'when', 'where', 'why', 'how', 'about', 'than', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'using', 'also', 'including', 'such', 'only', 'own', 'same', 'so', 'than', 'that', 'then', 'these', 'those', 'through', 'very', 'without', 'will', 'with', 'would', 'could', 'should', 'might'}
        for w in text.lower().split():
            w = w.strip('.,!?;:()[]{}"\'')
            if len(w) > 2 and w not in stopwords and w.isalpha():
                words.add(w)
    sport_flags = [w for w in words if w in sport_dict]
    return words, sport_flags

def score_semantic_relevance(sub_question, text, sport_dict, sport_bonus_weight=0.15):
    if not text:
        return 0.0
    sq_words, sq_sports = extract_keywords(sub_question, sport_dict)
    if not sq_words:
        return 0.5
    text_words, text_sports = extract_keywords(text, sport_dict)
    overlap = len(sq_words & text_words) / len(sq_words)
    sport_bonus = sport_bonus_weight if (sq_sports and text_sports) else 0.0
    return min(1.0, overlap + sport_bonus)

def filter_sql_results(sub_question, question_id, sql_by_id, refiner_map, cfg):
    refiner = refiner_map.get(question_id, {})
    if refiner.get("final_results"):
        return refiner["final_results"][:10], 1.0
    for rec in sql_by_id.get(question_id, []):
        output = rec.get("output") or {}
        if output.get("success"):
            rows = output.get("results") or []
            if rows:
                answers = []
                for row in rows:
                    if isinstance(row, (list, tuple)) and row:
                        val = str(row[0]).strip()
                    else:
                        val = str(row).strip()
                    if val and not re.match(r'^[a-z_]+$', val):
                        answers.append(val)
                if answers:
                    return answers[:10], 1.0
    return [], 0.0

def get_vector_results(sub_question, question_id, vector_map, cfg, sport_dict):
    vec = vector_map.get(question_id, {})
    retrieval = vec.get("retrieval_results", {})
    thresholds = cfg["thresholds"]
    top_k = thresholds.get("vector_top_k_per_type", 5)
    min_rel = thresholds.get("vector_relevance_min", 0.1)
    sport_bonus = thresholds.get("sport_bonus_weight", 0.15)

    type_weight = {
        "text": 1.0,
        "infobox": 1.2,
        "table": 1.1
    }

    candidates = []
    for result_type in ["text", "infobox", "table"]:
        bucket = retrieval.get(result_type, {}) or {}
        min_score = thresholds.get(f"vector_min_score_{result_type}", 0)
        items = bucket.get("results", [])
        typed_candidates = []
        for item in items:
            score = item.get("score", 0)
            if score < min_score:
                continue
            content = item.get("content", {})
            if result_type == "text":
                text = content.get("text", "")
            else:
                text = " | ".join(f"{k}: {v}" for k, v in content.items())
            rel = score_semantic_relevance(sub_question, text, sport_dict, sport_bonus)
            if rel >= min_rel:
                typed_candidates.append({"text": text, "relevance": rel, "type": result_type})
        typed_candidates.sort(key=lambda x: x["relevance"], reverse=True)
        candidates.extend(typed_candidates[:top_k])

    for c in candidates:
        c["score"] = c["relevance"] * type_weight.get(c["type"], 1.0)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return [c["text"] for c in candidates[:top_k]]

def call_llm(prompt, model, api_key, temperature=0):
    dashscope.api_key = api_key
    try:
        response = Generation.call(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            result_format="message"
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content.strip()
        else:
            print(f"LLM API error: {response.message}")
            return ""
    except Exception as e:
        print(f"LLM call failed: {e}")
        return ""

def answer_sub_question_with_llm(sub_q, filtered_answers, llm_config, prompts):
    if not filtered_answers:
        return "No relevant information found."
    prompt = prompts["answer_sub_question"].format(
        question=sub_q,
        retrieved_info=json.dumps(filtered_answers[:10], ensure_ascii=False)
    )
    return call_llm(prompt, llm_config["model"], llm_config["api_key"], llm_config["temperature"])

def fuse_answers_with_llm(original_question, sub_answers, combination, llm_config, prompts):
    combo_type = combination.get("type", "parallel")
    combo_desc = combination.get("description", "")
    prompt = prompts["fuse_answers"].format(
        original_question=original_question,
        combination_type=combo_type,
        combination_description=combo_desc,
        sub_answers=json.dumps(sub_answers, ensure_ascii=False, indent=2)
    )
    return call_llm(prompt, llm_config["model"], llm_config["api_key"], llm_config["temperature"])

def process_question(question_id, data_maps, cfg, sport_dict):
    decomp = data_maps["decomposer"].get(question_id, {})
    original_q = decomp.get("original_question", "")
    sub_questions = decomp.get("sub_questions", [])
    combination = decomp.get("combination", {})
    if not sub_questions:
        sub_questions = [original_q]

    llm_config = cfg.get("llm", {})
    prompts = cfg.get("prompts", {})
    has_llm = bool(llm_config.get("api_key") and prompts)

    sub_answers = []
    for sq in sub_questions:
        sql_ans, _ = filter_sql_results(sq, question_id, data_maps["sql_exec"],
                                         data_maps["refiner"], cfg)
        vec_ans = get_vector_results(sq, question_id, data_maps["vector"], cfg, sport_dict)
        merged = sql_ans if sql_ans else vec_ans
        if has_llm and merged:
            answer = answer_sub_question_with_llm(sq, merged, llm_config, prompts)
        elif merged:
            answer = "、".join(str(x) for x in merged[:5])
        else:
            answer = ""
        sub_answers.append(answer)

    if has_llm and len(sub_answers) > 1:
        final_answer = fuse_answers_with_llm(original_q, sub_answers, combination, llm_config, prompts)
    else:
        final_answer = sub_answers[0] if sub_answers else ""

    return {
        "question_id": question_id,
        "original_question": original_q,
        "sub_questions": sub_questions,
        "combination_type": combination.get("type", "parallel"),
        "sub_answers": sub_answers,
        "final_answer": final_answer
    }

def main():
    cfg = load_config()
    paths = cfg["paths"]
    print("使用方法:")
    print("  默认测试模式（前10条）：        python filter/filter.py")
    print("  测试模式（前10条）：            python filter/filter.py --mode test")
    print("  测试模式（前N条）：              python filter/filter.py --limit=N")
    print("  完整模式（全部问题）：           python filter/filter.py --mode full")
    limit = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--mode" and len(sys.argv) > 2:
            mode = sys.argv[2].lower()
            if mode == "full":
                limit = None
            elif mode == "test":
                limit = 10
        elif sys.argv[1].startswith("--limit="):
            limit = int(sys.argv[1].split("=")[1])
        else:
            limit = 10
    else:
        limit = 10

    print(f"运行模式: {'测试模式 (前{}条)'.format(limit) if limit else '完整模式'}")

    sport_dict = build_sport_dict(cfg)
    print(f"运动词典大小: {len(sport_dict)}")

    decomp_map = load_jsonl(paths["decomposer"], cfg)
    vector_map = load_jsonl(paths["vector_retrieval"], cfg)
    sql_records = load_jsonl(paths["sql_execution"], cfg, id_key=None)
    sql_by_id = {}
    for rec in sql_records.values():
        inp = rec.get("input") or {}
        qid = inp.get("id")
        if qid:
            sql_by_id.setdefault(qid, []).append(rec)

    refiner_map = load_refiner_json(paths["refiner"], cfg)

    data_maps = {
        "decomposer": decomp_map,
        "sql_exec": sql_by_id,
        "refiner": refiner_map,
        "vector": vector_map,
    }

    all_qids = sorted(set(decomp_map.keys()) | set(vector_map.keys()) | set(sql_by_id.keys()))
    if limit:
        all_qids = all_qids[:limit]

    results = []
    for qid in all_qids:
        res = process_question(qid, data_maps, cfg, sport_dict)
        results.append(res)

    out_path = resolve_path(cfg["output"]["filtered_results"], cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Filtered {len(results)} questions. Saved to {out_path}")

if __name__ == "__main__":
    main()