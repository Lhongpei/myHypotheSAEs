import os
import csv
import json
import concurrent.futures
import random # Import the random module
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import re
# --- Configuration ---
MAX_WORKERS = 32

# --- OpenAI Client Initialization (Unchanged) ---
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
if API_KEY is None:
    API_KEY = 'EMPTY'
client = OpenAI(api_key=API_KEY, base_url="http://0.0.0.0:1212/v1")

# --- Core Functions (Modified generate_profile) ---
def callChatGPT(prompt, model="Qwen/Qwen3-32B"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=None
    )
    return response.choices[0].message.content

def predict_discharge_location(idx, few_shot_profile_text,few_shot_labels, profile_text):
    prompt = (
        f"You are a clinical language model trained to predict hospital discharge outcomes based on patient profiles. \
        Given a narrative summary of the patient's condition and hospital stay, predict their likely discharge location."
        f"\n\n### Few-Shot Examples:\n"
    )
    for i in range(len(few_shot_profile_text)):
        prompt += (
            f"Patient Profile:\n{few_shot_profile_text[i]}\n"
            f"Discharge Location: {few_shot_labels[i]}\n\n"
        )
    prompt += (
        f"The patient you are currently analyzing has the following profile:\n"
        f"{profile_text}\n\n"
        f"Based on this profile, predict the most likely discharge location for this patient. "
        f"Your prediction should be only one of the following: 'home', 'other', or 'died', without any additional explanation\n"
    )
    profile_text = callChatGPT(prompt)
    return idx, profile_text

# --- Main Processing Function (Remains the same as the 'ordered_parallel' version) ---
def process_profile_jsonl_ordered_parallel(profile_jsonl_path, output_jsonl_path, few_shot_examples_jsonl, few_shot_labels_csv_path, few_shot_examples=5):


    
    profile_texts = []
    with open(profile_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            profile_texts.append(json.loads(line)['profile'])
    indices = list(range(len(profile_texts)))
    few_shot_examples_profile_texts = []
    few_shot_labels = []
    with open(few_shot_examples_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            few_shot_examples_profile_texts.append(json.loads(line)['profile'])
    with open(few_shot_labels_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            few_shot_labels.append(row[0])
        
    few_shot_labels = few_shot_labels[1:]      
    ordered_results = [None] * len(indices)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(predict_discharge_location, idx, few_shot_examples_profile_texts[:few_shot_examples], few_shot_labels[:few_shot_examples], profile_texts[idx]): idx
            for idx in indices
        }

        print(f"Submitting jobs to {MAX_WORKERS} workers...")
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(indices), desc="Generating Profiles"):
            try:
                original_idx, prediction = future.result()
                pattern = re.compile(r"<think>.*?</think>", re.DOTALL) 
                remove_think_prediction = re.sub(pattern, "", prediction)
                ordered_results[original_idx] = {
                    "idx": original_idx,
                    "prediction": remove_think_prediction,
                    "reasoning": prediction
                }
            except Exception as e:
                original_idx = future_to_idx[future]
                print(f"❌ Error processing patient {original_idx}: {e}")
                ordered_results[original_idx] = {"idx": original_idx, "profile": f"Error: {e}"}

    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for result in ordered_results:
            if result is not None:
                f_out.write(json.dumps(result) + '\n')
            else:
                print(f"❗ Warning: A profile was not generated or recorded for an index.")

    print(f"✅ All profiles have been generated and saved to {output_jsonl_path} in original order.")

if __name__ == '__main__':
    # process_csv_to_profile_jsonl_ordered_parallel('data/X_train.csv', 'data/train_profile_random_detail.jsonl')
    process_profile_jsonl_ordered_parallel(
        profile_jsonl_path='data/test_profile_random_detail.jsonl',
        # label_csv_path='data/y_test.csv',
        output_jsonl_path='test_predictions.jsonl',
        few_shot_examples_jsonl='data/train_profile_random_detail.jsonl',
        few_shot_labels_csv_path='data/y_train.csv',
        few_shot_examples=20
    )