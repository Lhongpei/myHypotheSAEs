import argparse
import json
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import concurrent.futures
from itertools import repeat
REASONING = True
USE_LABEL_DESCRIPTION = False  # Set to True to include label descriptions in the prompt
USE_STATISTICAL_RULES = True  # Set to True to include statistical rules in the prompt
RULES_TEXT = """
### Statistically-Derived Discharge Indicators (SAE + logistic regression)

### Instructions for LLM:

1. Read the patient profile and extract all relevant clinical details.

2. Initialize a score vector `S = [Home: 0, Other: 0]`.

3. For each hypothesis:  
   a. State whether it is **TRUE** or **FALSE** based on the profile.  
   b. If **TRUE**, update the vector:  
      - Positive coefficient → add the value to `Other`.  
      - Negative coefficient → add the **absolute value** to `Home`.  
   c. Report the incremental update:  
      `Hypothesis k: <TRUE|FALSE> → S = [Home: new_H, Other: new_O]`

4. After processing all hypotheses, output:  
   `Final scores: [Home: final_H, Other: final_O]`  
   `Answer: <Home|Other>`

Hypotheses & coefficients:
hypothesis : regression coef
  "The patient is aged 65 years or older.": 1.7209531174687576,
  "The text describes an emergency room admission at midnight as an 'emergency case' or 'observation admit' with a prolonged hospital stay (typically 6 days or longer) and mentions the patient being in the advanced age group.": 0.9828327771654995,
  "Patient age is in the 75+ years demographic (e.g., 75–84 or 85+ years).": 0.9535261819197546,
  "Mentions a prolonged hospital stay (length of stay significantly longer than typical for the admission type or age group).": 3.1117802339310283,
  "The text describes patients aged 85 years or older (85+ age group).": 2.0287697797930337,
  "The patient is between 65 and 84 years old, has a normal or borderline BMI, and was admitted at midnight (admit_hour: 0) through emergency or urgent care.": -0.019788714688514265,
  "The text describes a patient aged 75 years or older with stage 1 or stage 2 hypertension, specifically highlighting elevated systolic blood pressure (≥130 mmHg) as a key clinical concern.": 0.10294478111412264,
  "The text describes patients who are 75 years of age or older, have a BMI in the overweight or obese range, and exhibit isolated systolic hypertension or stage 1/2 hypertension, often with a low diastolic blood pressure.": -0.2065092365466639,
  "The text describes patients who are 65 years or older, have a BMI indicating overweight or obesity, and were admitted through the emergency room at midnight for an observation stay or a brief hospitalization.": 0.07940089710426444,
  "The text describes patients who are 75 years old or older, have a BMI of 25 or higher, and have blood pressure readings that indicate either hypertension or hypotension.": 0.046488665293829244,
  "Mentions a normal BMI or overweight BMI in the context of elderly patients (age 65+) with normal or only mildly elevated blood pressure readings.": 0.33753666502720286,
  "mentions a diastolic blood pressure value below 70 mmHg in elderly patients (age 75+).": -0.03428227755120213,
  "Mentions isolated systolic hypertension combined with advanced age (75+ years) and underweight or obese BMI, often in the context of increased fall risk, frailty, or cardiovascular complications.": 0.45462874852250784,
  "Mentions hospital admission at midnight (admit hour 0) in the context of an emergency or urgent admission, specifically for patients aged 50 or older with either normal or prehypertensive blood pressure and no prior records.": -0.16023763322878964,
  "The patient is admitted at midnight (admit hour: 0) via walk-in/self-referral or emergency status, and the hospitalization lasts for an extended duration (typically more than 4 days) without the presence of prior records.": 0.9803325004044705,
  "Mentions low diastolic blood pressure (diastolic value ≤ 70 mmHg) in the context of isolated systolic hypertension or as a potential indicator of hypoperfusion or hypotension risks.": 0.35995900231322675,
  "mentions underweight BMI (BMI < 18.5).": -0.06503879367043408,
  "The text describes an urgent admission at midnight via transfer from another hospital, with a prolonged length of stay (typically 4–16 days) and mentions the absence or limited prior documented healthcare records in the system.": 0.572777825553086,
  "The text mentions diastolic blood pressure values below 60 mmHg, indicating hypotension.": -0.6866200782446628,
  "The patient is male and has a BMI in the 'Normal' or 'Overweight' range, but not classified as 'Obese'.": -0.14685068831139125,
  "The text mentions the patient has a BMI categorizing them as overweight (25.0–29.9) and a blood pressure reading where the systolic value is elevated (≥140 mmHg).": -0.15746620559340282,
  "The text mentions that the patient was admitted via transfer from another hospital.": 1.2626575413987697,
  "The text describes the patient as underweight (BMI < 18.5) or includes blood pressure readings that indicate hypotension (systolic < 100 mmHg or diastolic < 60 mmHg).": 0.1698743245578804,
  "The text mentions a patient being admitted via walk-in/self-referral for an observation admit.": -2.363319080236203,
  "The presence of prehypertension or stage 1/2 hypertension combined with a lack of prior medical records (number_of_records: 0) or very limited prior healthcare engagement, suggesting newly identified or previously undiagnosed hypertension.": -0.21138213563588198,
  "The text mentions that the patient's blood pressure is within normal limits, with both systolic and diastolic values in the healthy range (typically systolic ≤120 mmHg and diastolic ≤84 mmHg).": -0.2959506895910465,
  "The text indicates that this is the patient's first documented healthcare encounter (number of prior records: 0).": -1.2499156300241574,
  "The text mentions the patient being Hispanic or Latino.": -0.47769975531327935,
  "Mentions obesity-related risk factors in a patient under 50 years of age.": -1.8784203387744642,
  "The text describes a patient under 50 years old with a normal BMI and normal or near-normal blood pressure readings, who was admitted urgently at midnight via physician referral for a brief hospital stay, with no or limited prior records.": -0.08892514489251206,
  "Patient is a young adult (<50 years old) with an emergency room admission at midnight.": -1.0813395909401726
"""
# System prompt for this evaluation
SYSTEM_PROMPT = "You are a senior clinical-reasoning assistant. Given a structured patient record *and* its correct discharge location, explain briefly why the label fits, then output exactly 'Answer: <Home|Other>' on a new line. "
DESCRIPTION = """
### Discharge Location Descriptions
- **Home**: Discharged to a private home without services.
- **Other**: Other cases including skilled nursing facility, rehab, chronic/long term acute care, psych facility, healthcare facility, home health care, assisted living, hospice, died and other facility, 
"""
# User prompt template (directly use profile)
INFERENCE_USER_PROMPT_TEMPLATE = """## Chain-of-Thought Instructions\n1. Start with **Thought:**\n2. Bullet 2–5 key clinical cues.\n3. End with **Answer: <Home|Other>**\nDo NOT mention any other label after the answer line.\n\n{profile}\n\nExplain briefly, then output the answer in the format 'Answer: <label>'."""

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
        system_prompt = system_prompt + RULES_TEXT + "\n\nPlease note the statistical rules above before answering the question. You need Absolutely follow the instruct to step by step calculate the value and then give me the final prediction!"#Also, you still need consider the original data distribution of the labels, which is [0.5, 0.2, 0.2, 0.1] for [Home, Facility-Based Care, Home with Services, Deceased/Hospice], which means that you can output the label with the highest number in data distribution if you are not sure about the answer.\n\n"
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
        "max_tokens": 25600
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
    x_dict = {1:'Other', 0:'Home'}
    true_labels_lower = [x_dict[x].lower() for x in true_labels] if type(true_labels[0]) is int else [x.lower() for x in true_labels]
    predicted_labels_lower = [x.lower().strip('.') for x in predictions]
    false_label = [i if true_labels_lower[i] != predicted_labels_lower[i] else None for i in range(len(true_labels_lower))]
    false_out = [model_outputs[i] if true_labels_lower[i] != predicted_labels_lower[i] else None for i in range(len(true_labels_lower))]
    false_instance_jsonl = {idx: {
        "profile": profiles[idx],
        "true_label": true_labels_lower[idx],
        "predicted_label": predicted_labels_lower[idx],
        "false_output": false_out[idx] if idx < len(false_out) else None,
        "patient_id": idx
    } for idx in false_label if idx is not None}
    all_instance_jsonl = {idx: {
        "profile": profiles[idx],
        "true_label": true_labels_lower[idx],
        "predicted_label": predicted_labels_lower[idx],
        "model_output": model_outputs[idx] if idx < len(model_outputs) else None,
        "patient_id": idx
    } for idx in range(len(profiles))}
    true_label = [i if true_labels_lower[i] == predicted_labels_lower[i] else None for i in range(len(true_labels_lower))]
    true_out = [model_outputs[i] if true_labels_lower[i] == predicted_labels_lower[i] else None for i in range(len(true_labels_lower))]
    true_instance_jsonl = {idx: {
        "profile": profiles[idx],
        "true_label": true_labels_lower[idx],
        "predicted_label": predicted_labels_lower[idx],
        "model_output": true_out[idx] if idx < len(true_out) else None,
        "patient_id": idx
    } for idx in true_label if idx is not None}
    with open('true_instances.jsonl', 'w', encoding='utf-8') as f:
        for item in true_instance_jsonl.values():
            f.write(json.dumps(item) + '\n')
    with open('all_instances.jsonl', 'w', encoding='utf-8') as f:
        for item in all_instance_jsonl.values():
            f.write(json.dumps(item) + '\n')
    # 保存错误实例到 false_instances.jsonl
    with open('false_instances.jsonl', 'w', encoding='utf-8') as f:
        for item in false_instance_jsonl.values():
            f.write(json.dumps(item) + '\n')
            
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
    parser.add_argument("--num_workers", type=int, default=128,
                        help="Number of parallel workers.")
    # >>> 新增参数
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of repeated runs for statistics.")
    # <<<

    args = parser.parse_args()

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
