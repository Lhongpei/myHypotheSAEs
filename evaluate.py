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
      - Positive coefficient → add the value to `Home`.  
      - Negative coefficient → add the **absolute value** to `Other`.  
   c. Report the incremental update:  
      `Hypothesis k: <TRUE|FALSE> → S = [Home: new_H, Other: new_O]`

4. After processing all hypotheses, output:  
   `Final scores: [Home: final_H, Other: final_O]`  
   `Answer: <Home|Other>`

Hypotheses & coefficients:
hypothesis : regression coef
 "Mentions a patient under the age of 22 with recorded BMI and/or blood pressure values indicating overweight or hypertension, or with missing anthropometric data but a short hospital stay suggesting an acute or complex condition.",1.0333432597224388
 "Mentions missing systolic and diastolic blood pressure, height, weight, or BMI data in young adult patients (age 18–27) with no prior medical records and urgent or emergency admission",1.266786365230042
 "The patient is between 18 and 24 years old and has zero or very limited prior medical records, indicating new engagement with the healthcare system or underutilization of preventive services.",-0.2147439450734255
 "The patient is an 18-22 year old individual with limited or no prior medical records and incomplete or missing vital signs such as BMI, systolic and diastolic blood pressure, height, and weight.",-0.3280718265621745
 "The patient has no prior medical records in the system, indicating a lack of documented healthcare engagement prior to this visit.",0.3846947206071551
 "Mentions a BMI that is either normal or classified as overweight, but does not indicate underweight or severely obese status.",0.1746629296692695
 "Mentions admission through a physician referral pathway",-0.18221917628277046
 "mentions missing systolic and diastolic blood pressure values in the context of overweight or obesity and acute admission",0.31407432617727776
 "Mentions the absence of systolic and/or diastolic blood pressure readings in the patient's medical records.",-0.2658240107850307
 "mentions no prior medical records and absence of measurable vital signs such as blood pressure, BMI, height, and weight",0.627464496945624
 "Mentions both systolic and diastolic blood pressure values within a single blood pressure reading (e.g., '120/48 mmHg')",-0.10564663154608032
 "Lacks documentation of systolic and diastolic blood pressure, height, weight, and BMI",-0.17210103242705377
 "The text mentions the absence of recorded systolic and diastolic blood pressure, BMI, height, and weight data.",0.008862601034864814
 "Absence of recorded BMI, blood pressure, height, and weight metrics in the patient's clinical documentation",-0.8158571802451184
 "Mentions a hospital transfer admission (e.g., 'admitted following transfer from another hospital', 'transferred from another hospital')",-0.5127082895144833
 "Mentions prolonged hospitalization (length of stay ≥ 4 days) in the context of urgent or emergency admission, often highlighting unresolved clinical complexities, barriers to discharge, or extended recovery needs.",-0.6804339484511772
 "The text mentions the patient being admitted via the emergency room and describes them as elderly (age 82 or older) with no or limited prior healthcare records, unrecorded blood pressure, BMI, and anthropometric data",0.2677036723916091
 "mentions an extended length of stay (typically 10 days or more)",-1.3411972967472179
 "Mentions an age of 80 years or older",-1.5195895797149535
 "The patient is 91 years old or older and is admitted to the hospital, with a focus on geriatric concerns such as frailty, falls, and age-related health risks.",-2.0670867237976314
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
    predicted_labels_lower_1 = [x.lower().strip('.') for x in predictions]
    predicted_labels_lower = ['home' if x == 'home' else 'other' for x in predicted_labels_lower_1]
    true_labels_lower = [ 'home' if x == 'home' else 'other' for x in true_labels_lower]
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
    metrics = {"accuracy": acc}
    for label, v in report.items():
        if isinstance(v, dict):  # skip 'accuracy', 'macro avg', 'weighted avg'
            metrics[f"{label}_precision"] = v["precision"]
            metrics[f"{label}_recall"] = v["recall"]
            metrics[f"{label}_f1"] = v["f1-score"]
    cm = confusion_matrix(true_labels_lower, predicted_labels_lower, labels=all_labels)
    metrics['cm'] = cm 
    metrics['all_labels'] = all_labels 
    return metrics
def print_summary_table(df_metrics: pd.DataFrame, all_labels) -> None:
    """
    以 classification_report 
    """
    cols = [c for c in df_metrics.columns
            if c.endswith(("_precision", "_recall", "_f1"))]

    label_rows = {}
    for c in cols:
        *label, metric = c.rsplit("_", 1)
        label = "_".join(label)
        if label not in label_rows:
            label_rows[label] = {}
        label_rows[label][metric] = c

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
