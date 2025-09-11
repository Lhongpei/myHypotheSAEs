import argparse
import json
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import concurrent.futures
from itertools import repeat
REASONING = False
USE_LABEL_DESCRIPTION = True  # Set to True to include label descriptions in the prompt
USE_STATISTICAL_RULES = False  # Set to True to include statistical rules in the prompt
RULES_TEXT = """
### Statistically-Derived Discharge Indicators (SAE + logistic regression)
Reference these steps to calculate the prior probabilities for each discharge location:

# 1. Read the patient profile.  
# 2. For each hypothesis below, check if it is relevant to the patient profile.
#    - If the hypothesis is TRUE, add its coefficient to the corresponding class.
#    - If the hypothesis is FALSE, do not add its coefficient.
# 3. Compute logits for the 4 classes:  
#    - Start at [0.0, 0.0, 0.0, 0.0] for [Home, Facility Based Care, Home with Services, Deceased/Hospice].
#    - If a hypothesis is TRUE, add its coefficient to the corresponding class.  
# 4. Softmax the logits to probabilities.
# 5. Output exactly one line:  
#    Answer: <highest-probability-label>

Hypotheses & coefficients:
H0  +0.67  age ≥ 80 yr  → Deceased/Hospice  
H1  +0.58  no BP/BMI/ht/wt recorded  → Deceased/Hospice  
H2  +0.88  physician-referral + LOS long  → Deceased/Hospice  
H3  +0.66  isolated systolic HTN ≥140  → Deceased/Hospice  
H4  +1.01  missing vitals + no prior records  → Deceased/Hospice  
H5  -0.97  first-time encounter  → Deceased/Hospice  
H6  -2.34  same-day surgery + vitals documented  → Deceased/Hospice  
H7  -16.7  young adult 18-22  → Deceased/Hospice  
H8  +2.21  same-day surgery admit  → Home  
H9  +0.50  elective admit via physician referral  → Home  
H10 +0.10  Stage 1 HTN  → Home  
H11 +0.43  obese + hypertensive  → Home  
H12 +0.66  vitals fully documented  → Home  
H13 -1.44  same-day surgery + no prior records  → Home  
H14 -0.57  urgent admit 2-3 d + missing vitals  → Home  
H15 -0.85  emergency admit 4+ d + missing vitals  → Home  
H16 -0.92  age ≥81  → Home  
H17 -1.68  LOS ≥10 d + missing vitals  → Home  
H18 +1.24  no prior records  → Home_with_Services  
H19 +0.65  young adult 18-22 + BMI ≥25 + missing BP  → Home_with_Services  
H20 -0.58  young adult 18-24 + limited records  → Home_with_Services  
H21 -0.67  vitals documented normal  → Home_with_Services  
H22 -0.86  missing BP values  → Home_with_Services  
H23 -0.92  age ≥81  → Home_with_Services  
H24 -1.68  LOS ≥10 d + missing vitals  → Home_with_Services  
H25 -22.4  91-92 yr White + ER + emergency  → Home_with_Services  
H26 +1.46  LOS ≥4 d  → Facility_Based_Care  
H27 +1.06  age ≥81 + ER admit + no prior records  → Facility_Based_Care  
H28 +0.61  admitted for observation post-SNF  → Facility_Based_Care  
H29 +0.66  divorced + prolonged/urgent admit + missing vitals  → Facility_Based_Care  
H30 -0.81  diastolic optimal + systolic normal  → Facility_Based_Care  
H31 -0.92  age ≥81  → Facility_Based_Care  
H32 -1.68  LOS ≥10 d + missing vitals  → Facility_Based_Care  
+ means increase probability, - means decrease probability.
"""
# System prompt for this evaluation
SYSTEM_PROMPT = "You are a senior clinical-reasoning assistant. Given a structured patient record *and* its correct discharge location, explain briefly why the label fits, then output exactly 'Answer: <Home|Facility-Based Care|Home with Services|Deceased / Hospice>' on a new line. "
DESCRIPTION = """
### Discharge Location Descriptions
- **Home**: Discharged to a private home without services.
- **Facility-Based Care**: Transferred to another inpatient facility (such as skilled nursing facility, rehab, chronic/long term acute care, psych facility, healthcare facility and other facility).
- **Home with Services**: Discharged to a home setting with planned services (such as home health care and assisted living).
- **Deceased / Hospice**: The patient is on an end-of-life pathway (such as died and hospice).
"""
# User prompt template (directly use profile)
INFERENCE_USER_PROMPT_TEMPLATE = """## Chain-of-Thought Instructions\n1. Start with **Thought:**\n2. Bullet 2–5 key clinical cues.\n3. End with **Answer: <Home|Facility-Based Care|Home with Services|Deceased / Hospice>**\nDo NOT mention any other label after the answer line.\n\n{profile}\n\nExplain briefly, then output the answer in the format 'Answer: <label>'."""

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def query_model(api_url, model_path, profile):
    prompt = INFERENCE_USER_PROMPT_TEMPLATE.format(profile=profile)
    system_prompt = SYSTEM_PROMPT
    if not REASONING:
        prompt = prompt + '/no_think'
    if USE_STATISTICAL_RULES:
        # prompt = prompt + '\n\n' + RULES_TEXT + '\n\n' + "Please note the statistical rules above before answering the question.\n\n"
        system_prompt = system_prompt + RULES_TEXT + "\n\nPlease note the statistical rules above before answering the question. It's only a reference, you still need combine your knowleadge to judge the final prediction. "#Also, you still need consider the original data distribution of the labels, which is [0.5, 0.2, 0.2, 0.1] for [Home, Facility-Based Care, Home with Services, Deceased/Hospice], which means that you can output the label with the highest number in data distribution if you are not sure about the answer.\n\n"
        # prompt = prompt + '\n\n' + DESCRIPTION + '\n\n' + "Please note the discharge location descriptions above before answering the question.\n\n"
    if USE_LABEL_DESCRIPTION:
        system_prompt = system_prompt + DESCRIPTION + "\n\nPlease note the discharge location descriptions above before answering the question.\n\n"

    payload = {
        "model": model_path,
        "messages": [
            {"role": "system", "content":system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "top_p": 0.7,
        "max_tokens": 12800
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"API request failed for a record: {e}")
        return "request_failed"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Failed to parse API response: {getattr(response, 'text', '')}. Error: {e}")
        return "parsing_failed"

def parse_model_output(output):
    if output is None or output in ["request_failed", "parsing_failed"]:
        return output if output is not None else "parsing_failed"
    for line in output.strip().split('\n'):
        if line.lower().startswith("answer:"):
            clean_line = line.replace('*', '').strip()
            return clean_line.split(":", 1)[1].strip()
    # Fallback: look for any of the valid labels in output
    flag = False
    for word in ["Home", "Facility-Based Care", "Home with Services", "Deceased"]:
        
        if word.lower() in output.lower():
            flag = True
            if '<' in word or '>' in word:
                word = word.replace('<', '').replace('>', '').strip()
            return word
    if not flag:
        print(f"Warning: No valid label found in output: {output}")
    return "parsing_failed"

def run_once(args):
    """
    执行一次完整推理 & 评估，返回关键指标（dict）。
    """
    data = load_jsonl(args.test_jsonl_path)
    profiles = [item['profile'] for item in data]
    true_labels = [item['label'] for item in data]
    idxs = [item.get('idx', i) for i, item in enumerate(data)]

    model_outputs = []
    print(f"Starting inference on {len(profiles)} records with {args.num_workers} parallel workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results_iterator = executor.map(query_model, repeat(args.api_url),
                                        repeat(args.model_path), profiles)
        for model_output in tqdm(results_iterator, total=len(profiles)):
            model_outputs.append(model_output)

    predictions = [parse_model_output(out) for out in model_outputs]
    predictions = [p.replace('<', '').replace('>', '').strip() for p in predictions]

    true_labels_lower = [x.lower() for x in true_labels]
    predicted_labels_lower = [x.lower() for x in predictions]
    all_labels = sorted(list(set(true_labels_lower) | set(predicted_labels_lower)))
    # 计算指标
    acc = accuracy_score(true_labels_lower, predicted_labels_lower)
    report = classification_report(true_labels_lower, predicted_labels_lower,
                                   output_dict=True, zero_division=0)

    # 把所有标签的 f1-score、precision、recall 打平
    metrics = {"accuracy": acc}
    for label, v in report.items():
        if isinstance(v, dict):  # skip 'accuracy', 'macro avg', 'weighted avg'
            metrics[f"{label}_precision"] = v["precision"]
            metrics[f"{label}_recall"] = v["recall"]
            metrics[f"{label}_f1"] = v["f1-score"]
    cm = confusion_matrix(true_labels_lower, predicted_labels_lower, labels=all_labels)
    metrics['cm'] = cm 
    metrics['all_labels'] = all_labels  # 保存所有标签以便后续打印
    return metrics
def print_summary_table(df_metrics: pd.DataFrame, all_labels) -> None:
    """
    以 classification_report 风格打印多轮均值±方差表
    """
    # 只保留需要的列（precision / recall / f1）
    cols = [c for c in df_metrics.columns
            if c.endswith(("_precision", "_recall", "_f1"))]

    # 按 label 聚合
    label_rows = {}
    for c in cols:
        *label, metric = c.rsplit("_", 1)
        label = "_".join(label)
        if label not in label_rows:
            label_rows[label] = {}
        label_rows[label][metric] = c

    # 按 label 排序，保持可读性
    ordered_labels = [
        "deceased / hospice",
        "facility-based care",
        "home",
        "home with services",
        "macro avg",
        "weighted avg"
    ]
    # 若存在未覆盖的 label，追加到后面
    for lab in label_rows:
        if lab not in ordered_labels:
            ordered_labels.append(lab)

    # 打印表头
    print("\n=== Summary over {} runs ===".format(len(df_metrics)))
    print("{:<20} {:<15} {:<15} {:<15}".format(
        "", "precision", "recall", "f1-score"))
    for label in ordered_labels:
        if label not in label_rows:
            continue
        d = label_rows[label]
        prec = df_metrics[d["precision"]].mean()
        prec_std = df_metrics[d["precision"]].std()
        rec = df_metrics[d["recall"]].mean()
        rec_std = df_metrics[d["recall"]].std()
        f1 = df_metrics[d["f1"]].mean()
        f1_std = df_metrics[d["f1"]].std()
        print("{:<20} {:<15} {:<15} {:<15}".format(
            label,
            f"{prec:.4f}±{prec_std:.4f}",
            f"{rec:.4f}±{rec_std:.4f}",
            f"{f1:.4f}±{f1_std:.4f}"
        ))
    acc_mean = df_metrics["accuracy"].mean()
    acc_std  = df_metrics["accuracy"].std()
    print("─" * 65)
    print("{:<20} {:<15} {:<15} {:<15}".format(
        "accuracy", "", "", f"{acc_mean:.4f}±{acc_std:.4f}"))
    
    cms = df_metrics["cm"].tolist()                # 每轮的 4×4 np.array
    avg_cm = np.round(np.mean(cms, axis=0)).astype(int)
    print("\n--- Average Confusion Matrix over {} runs ---".format(len(df_metrics)))
    print(pd.DataFrame(avg_cm,
                       index=all_labels,
                          columns=all_labels).to_string(index=True, header=True))
# -------------------------------
# 主函数
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a model using vLLM API on test_all_info.jsonl.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model for API request.")
    parser.add_argument("--test_jsonl_path", type=str,
                        default="../data/full_data/test_all_info.jsonl",
                        help="Path to test_all_info.jsonl.")
    parser.add_argument("--api_url", type=str,
                        default="http://localhost:8000/v1/chat/completions",
                        help="vLLM API endpoint.")
    parser.add_argument("--output_csv", type=str,
                        default="predictions_testallinfo.csv",
                        help="Path to save predictions CSV.")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Number of parallel workers.")
    # >>> 新增参数
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of repeated runs for statistics.")
    # <<<

    args = parser.parse_args()

    if args.runs <= 1:
        # 兼容旧用法：只跑一次
        data = load_jsonl(args.test_jsonl_path)
        profiles = [item['profile'] for item in data]
        true_labels = [item['label'] for item in data]
        idxs = [item.get('idx', i) for i, item in enumerate(data)]

        model_outputs = []
        print(f"Starting inference on {len(profiles)} records with {args.num_workers} parallel workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            results_iterator = executor.map(query_model, repeat(args.api_url),
                                            repeat(args.model_path), profiles)
            for model_output in tqdm(results_iterator, total=len(profiles)):
                model_outputs.append(model_output)

        predictions = [parse_model_output(out) for out in model_outputs]
        predictions = [p.replace('<', '').replace('>', '').strip() for p in predictions]

        results_df = pd.DataFrame({
            'idx': idxs,
            'profile': profiles,
            'predicted_discharge_location': predictions,
            'true_discharge_location': true_labels
        })
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nPredictions saved to {args.output_csv}")

        true_labels_lower = [x.lower() for x in true_labels]
        predicted_labels_lower = [x.lower() for x in predictions]
        all_labels = sorted(list(set(true_labels_lower) | set(predicted_labels_lower)))
        print(set(predicted_labels_lower))
        print(set(true_labels_lower))
        print("\n--- Evaluation Metrics ---")
        print(f"Accuracy: {accuracy_score(true_labels_lower, predicted_labels_lower):.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels_lower, predicted_labels_lower,
                                  labels=all_labels, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels_lower, predicted_labels_lower,
                              labels=all_labels))
        print("--------------------------")
    else:
        # 多次实验
        all_metrics = []
        for run_id in range(1, args.runs + 1):
            print(f"\n========== Run {run_id}/{args.runs} ==========")
            # 若想每次保存不同文件名，可在此处动态修改 args.output_csv
            m = run_once(args)
            m["run_id"] = run_id
            all_metrics.append(m)

        # 统计
        df_metrics = pd.DataFrame(all_metrics).set_index("run_id")
        prefix_out = args.output_csv.rsplit('.', 1)[0]
        args.stats_csv = f"{prefix_out}_stats_run_{args.runs}.csv"
        df_metrics.to_csv(args.stats_csv)
        print(f"\nPer-run metrics saved to {args.stats_csv}")

        print_summary_table(df_metrics, m["all_labels"])


if __name__ == "__main__":
    main()
