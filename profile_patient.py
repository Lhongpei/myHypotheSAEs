import os
import csv
import json
import concurrent.futures
import random # Import the random module

from tqdm import tqdm

# --- Configuration ---
MAX_WORKERS = 32
from dotenv import load_dotenv
from openai import OpenAI
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

def generate_profile(patient_with_idx):
    idx, patient = patient_with_idx

    # Collect all patient attributes into a dictionary
    patient_data_raw = { # Renamed to avoid confusion with shuffled version
        "gender": patient['gender'],
        "race": patient['race'],
        "age": patient['age'],
        "height": patient['height'],
        "weight": patient['weight'],
        "bmi": patient['bmi'],
        "marital_status": patient['marital_status'],
        "insurance": patient['insurance'],
        "language": patient['language'],
        "systolic_bp": patient['systolic_bp'],
        "diastolic_bp": patient['diastolic_bp'],
        "admit_type": patient['admit_type'],
        "admit_location": patient['admit_location'],
        "number_of_records": patient['number_of_records'],
        "length_of_stay": patient['length_of_stay'],
        "marital_status": patient['marital_status'],
        "discharge_time": patient['discharge_time'],
        "admit_hour": patient['admit_hour'],
        "age_group": patient['age_group']
    }

    # Add BMI interpretation
    if patient_data_raw['bmi'] == 'unknown':
        bmi_status = 'unknown'
    else:
        bmi_status = 'Underweight' if float(patient_data_raw['bmi']) < 18.5 else \
                    'Normal' if float(patient_data_raw['bmi']) < 25 else \
                    'Overweight' if float(patient_data_raw['bmi']) < 30 else 'Obese'
    patient_data_raw['bmi_status'] = bmi_status

    # --- NEW: Randomize the order of patient_attributes for the JSON output ---
    # Get all keys
    keys = list(patient_data_raw.keys())
    # Shuffle the keys randomly
    random.shuffle(keys)

    # Create a new dictionary with the shuffled order of keys
    patient_attributes_shuffled = {key: patient_data_raw[key] for key in keys}

    # Construct the prompt with raw data and clear instructions for randomization
    prompt = (
        f"You are an experienced healthcare data analyst specializing in creating detailed patient profiles.\n"
        f"Generate a comprehensive patient profile based on the following demographic and medical record data. "
        f"**Present the information in a natural, flowing narrative, varying the order of details and analysis points to create diverse profiles.** "
        f"Do not simply list the information in the order it is provided below. Focus on creating a professional and insightful report.\n\n"
        f"### Patient Raw Data (Order of attributes is randomized for diversity):\n"
        f"{json.dumps(patient_attributes_shuffled, indent=2)}\n\n" # Use the shuffled dictionary here
        f"### Narrative Patient Profile Analysis Requirements:\n"
        f"Synthesize the patient's health profile into a single, coherent narrative paragraph. Begin with a concise overview of their current health status, including BMI assessment and blood pressure analysis. Seamlessly integrate an analysis of their healthcare utilization pattern from this visit. Proceed to identify potential health risks grounded in the available data. Conclude with personalized health interventions and preventive measures. Maintain a strictly professional, objective tone, using data-driven insights throughout. Ensure the narrative flows logically without bullet points, subsections, or fragmented statements."
        f"### Patient Profile Analysis:"
    )
    # prompt = (
    #     f"You are a medical data analysis system using a Sparse Autoencoder (SAE) for patient representation.\n"
    #     f"Your task is to objectively describe the key clinical and demographic features of a patient based solely on the provided input data.\n\n"
    #     f"### Input Patient Vector (attributes randomized):\n"
    #     f"{json.dumps(patient_attributes_shuffled, indent=2)}\n\n"
    #     f"### Output Requirements:\n"
    #     f"1. Provide a structured, concise summary of the patient's attributes.\n"
    #     f"2. Include numeric interpretations where relevant.\n"
    #     f"3. Highlight clinically relevant indicators and potential risk factors.\n"
    #     f"4. Use precise, technical language with no repetition or storytelling.\n"
    #     f"5. The output must resemble a factual medical summary, not a narrative.\n\n"
    #     f"### Patient Profile (Objective SAE-style Output):"
    # )
    profile_text = callChatGPT(prompt)
    return idx, profile_text

# --- Main Processing Function (Remains the same as the 'ordered_parallel' version) ---
def process_csv_to_profile_jsonl_ordered_parallel(csv_path, output_jsonl_path):
    patients_with_indices = []
    with open(csv_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for idx, row in enumerate(reader):
            patients_with_indices.append((idx, row))
    print(f"Loaded {len(patients_with_indices)} patient records from {csv_path}.")

    ordered_results = [None] * len(patients_with_indices)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(generate_profile, patient_with_idx): original_idx
            for original_idx, patient_with_idx in enumerate(patients_with_indices)
        }

        print(f"Submitting jobs to {MAX_WORKERS} workers...")
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(patients_with_indices), desc="Generating Profiles"):
            try:
                original_idx, profile_text = future.result()
                ordered_results[original_idx] = {
                    "idx": original_idx,
                    "profile": profile_text
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
    process_csv_to_profile_jsonl_ordered_parallel('toy_data/X_test.csv', 'toy_data/test_profile.jsonl')