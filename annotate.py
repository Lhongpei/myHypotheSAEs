
# response = callChatGPT(prompt, model=model).split('>')[-1].strip().lower()
# --- Configuration ---
import json, os, csv
from dotenv import load_dotenv
from openai import OpenAI
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# ------------------ 配置 ------------------
MAX_WORKERS   = 128
REASONING     = False
MODEL_NAME    = "Qwen/Qwen3-32B"
REPEAT_TIMES  = 3          # 投票次数
DATASET_TYPE = 'test'
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
client  = OpenAI(api_key=API_KEY, base_url="http://0.0.0.0:1212/v1")

DATA_DIR  = "toy_data"
TRAIN_P   = os.path.join(DATA_DIR, "train_profile_toy.jsonl")
TEST_P    = os.path.join(DATA_DIR, "test_profile_toy.jsonl")
HYP_P     = "hypotheses.json"
OUT_CSV   = os.path.join(DATA_DIR, f"annotated_{DATASET_TYPE}.csv")

# ------------------ 核心函数 ------------------
def callChatGPT(prompt: str, model=MODEL_NAME) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

def annotate_profile(profile: dict, hypothesis: str, repeat_times: int = REPEAT_TIMES) -> bool | None:
    answers = []
    prompt_template = (
        "You are an expert in evaluating the truthfulness of statements based on user profiles.\n"
        "Given a user profile and a hypothesis, determine whether the hypothesis is true or false according to the profile.\n"
        "You need give me the evidence of your decision to support your answer.\n"
        "Finally, provide your answer as either \"True\" or \"False\".\n"
        f"User Profile: {json.dumps(profile, ensure_ascii=False)}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:\n"
    )
    if not REASONING:
        prompt_template += "/no_think"
    
    for _ in range(repeat_times):
        raw = callChatGPT(prompt_template).split('>')[-1].strip().lower()
        # print("Current Hypothesis:", hypothesis)    
        # print(f"Debug: raw response: {raw}")
        if "true" in raw:
            answers.append(True)
        elif "false" in raw:
            answers.append(False)
        else:
            answers.append(None)

    counter = Counter(answers)
    # 多数投票
    most, cnt = counter.most_common(1)[0]
    if cnt > repeat_times // 2:
        return most
    # 平票：True > False > None
    for cand in (True, False, None):
        if cand in counter:
            return cand
    return None

# ------------------ 读数据 ------------------
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_profiles = load_jsonl(TRAIN_P)
test_profiles  = load_jsonl(TEST_P)
with open(HYP_P, encoding="utf-8") as f:
    hypotheses = json.load(f)          # 假设是 {"0": "...", "1": "...", ...}

profiles = train_profiles if DATASET_TYPE == 'train' else test_profiles
h_keys = sorted(hypotheses.keys(), key=lambda x: int(x))  # 假设 ID 是数字字符串

# 生成任务列表：[(profile_id, profile, hyp_id, hyp), ...]
tasks = [(prof["idx"], prof['profile'], hid, hypotheses[hid])
         for prof in profiles
         for hid in h_keys]

# ------------------ 并行执行 ------------------
results = {}                       # (profile_id, hyp_id) -> verdict
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_task = {
        executor.submit(annotate_profile, prof, hyp): (pid, hid)
        for pid, prof, hid, hyp in tasks
    }
    for future in tqdm(as_completed(future_to_task),
                       total=len(future_to_task),
                       desc="Evaluating"):
        pid, hid = future_to_task[future]
        try:
            verdict = future.result()
        except Exception as e:
            print(f"Error on profile {pid}, hyp {hid}: {e}")
            verdict = None
        results[(pid, hid)] = verdict

# ------------------ 整理并写出 CSV ------------------

# ---------- 写 CSV ----------
idxs = sorted({p["idx"] for p in profiles})
header = ["idx"] + h_keys
rows = []
for idx in idxs:
    rows.append({"idx": idx, **{hid: results.get((idx, hid)) for hid in h_keys}})

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ 完成！结果已写入 {OUT_CSV}（共 {len(rows)} × {len(h_keys)} 个单元格）\n")

# ---------- 打印前 5 行 Markdown ----------
print("| " + " | ".join(header) + " |")
print("| " + " | ".join(["---"] * len(header)) + " |")
for row in rows[:5]:
    print("| " + " | ".join(str(row[h]) if row[h] is not None else "" for h in header) + " |")