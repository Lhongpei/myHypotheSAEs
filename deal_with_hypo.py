import json
import re
import os
from openai import OpenAI
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
def parse_interpretation_string(interpretation_str):
    """
    Parses the interpretation string to extract individual interpretation statements
    and their classification types.

    Args:
        interpretation_str (str): The interpretation string from the JSONL data.

    Returns:
        list[dict]: A list of dictionaries, where each dict has 'statement' and 'type'.
                    Example: [{'statement': 'mentions advanced age...', 'type': 'Deceased / Hospice_vs_rest'}, ...]
    """
    interpretations = []
    # Regular expression to find statements, activation, and type
    # It looks for:
    # - a statement (anything before ' (activation=')
    # - activation value
    # - type (inside type=[...])
    pattern = r"^- (.+?) \(activation=[\d.]+, type=\[(.+?)\]\)"
    lines = interpretation_str.strip().split('\n')

    # Skip the first line "This text likely involves:"
    for line in lines[1:]:
        match = re.match(pattern, line.strip())
        if match:
            statement = match.group(1).strip()
            # Clean up the statement if it starts with "The text mentions that " etc.
            if statement.lower().startswith("the text mentions that "):
                statement = statement[len("the text mentions that "):].strip()
            elif statement.lower().startswith("the text mentions "):
                statement = statement[len("the text mentions "):].strip()
            elif statement.lower().startswith("mentions "):
                statement = statement[len("mentions "):].strip()
            
            classification_type = match.group(2).strip()
            interpretations.append({
                "statement": statement,
                "type": classification_type,
                "activation": float(re.search(r'activation=([\d.]+)', line).group(1))
            })
    return interpretations

def create_interpretation_classification_dict(jsonl_file_path):
    """
    Reads a JSONL file and creates a dictionary mapping each
    'text' ID to a list of its interpretations and their types.

    Args:
        jsonl_file_path (str): Path to the JSONL file.

    Returns:
        dict: A dictionary where keys are 'text' IDs (e.g., 0, 1, 2) and
              values are lists of dictionaries, each containing 'statement' and 'type'.
    """
    interpretation_data = {}
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                text_id = entry["text"]
                interpretation_str = entry["interpretation"]
                
                parsed_interpretations = parse_interpretation_string(interpretation_str)
                interpretation_data[text_id] = parsed_interpretations
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
            except KeyError as e:
                print(f"Missing key in JSON entry: {line.strip()}. Error: {e}")
    return interpretation_data
def create_interpretation_classification_dict(jsonl_file_path):
    """
    Reads a JSONL file and creates a dictionary mapping each
    'text' ID to a list of its interpretations and their types.

    Args:
        jsonl_file_path (str): Path to the JSONL file.

    Returns:
        dict: A dictionary where keys are 'text' IDs (e.g., 0, 1, 2) and
              values are lists of dictionaries, each containing 'statement' and 'type'.
    """
    interpretation_data = {}
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                text_id = entry["text"]
                interpretation_str = entry["interpretation"]
                
                parsed_interpretations = parse_interpretation_string(interpretation_str)
                interpretation_data[text_id] = parsed_interpretations
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
            except KeyError as e:
                print(f"Missing key in JSON entry: {line.strip()}. Error: {e}")
    return interpretation_data

def _call_llm_for_single_interpretation(client, model, original_text, interpretation_item):
    """
    Helper function to call the LLM for a single interpretation statement.
    Designed to be run in parallel.
    """
    statement = interpretation_item['statement']
    classification_type = interpretation_item['type']

    system_prompt = (
        "You are an expert text analyst specializing in medical text interpretation and classification. "
        "Your task is to analyze a single 'Interpretation Statement' based ONLY on the 'Original Text'. "
        "Perform two tasks:\n"
        "1.  **Truthfulness Detection:** Determine if the statement is factually true and fully supported by the 'Original Text'. Do not use external knowledge.\n"
        "2.  **Utility Description:** If the statement is true, provide a concise, natural language description of *how* this specific statement would be useful for classifying the text under its associated 'Classification Type'. Focus on the informational value of the statement for that classification. If the statement is false, the utility description should be empty.\n\n"
        "Respond with a JSON object containing the following keys:\n"
        "- 'statement': (string) The original interpretation statement.\n"
        "- 'classification_type': (string) The classification type associated with the statement.\n"
        "- 'is_true': (boolean) True if the statement is fully supported by the Original Text, false otherwise.\n"
        "- 'reason': (string) A concise explanation for your 'is_true' decision, referencing the Original Text.\n"
        "- 'confidence': (float) Your confidence level in the 'is_true' decision, from 0.0 to 1.0.\n"
        "- 'utility_description': (string) A natural language description of the statement's utility for its classification type. Empty if 'is_true' is false."
    )

    user_prompt = f"""
                Original Text:
                ---
                {original_text}
                ---

                Interpretation Statement: "{statement}"
                Classification Type: "{classification_type}"

                Is the Interpretation Statement true based *only* on the Original Text? Provide your response in JSON format as described.
                """

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # Aim for deterministic, factual responses
        )
        llm_output = response.choices[0].message.content
        parsed_response = json.loads(llm_output)
        # Ensure the parsed response contains all expected keys, sometimes LLM might omit a key
        # if the utility_description is empty etc.
        parsed_response['statement'] = statement # Ensure original statement is preserved
        parsed_response['classification_type'] = classification_type # Ensure original type is preserved
        
        # Default empty string if utility_description is missing or not true
        if not parsed_response.get('is_true', False):
            parsed_response['utility_description'] = parsed_response.get('utility_description', '')
        
        return parsed_response
    except json.JSONDecodeError as e:
        print(f"Error decoding LLM's JSON response for statement '{statement}': {e}\nRaw LLM output: {llm_output}")
        return {
            "statement": statement,
            "classification_type": classification_type,
            "is_true": False,
            "reason": "JSON decoding error from LLM",
            "confidence": 0.0,
            "utility_description": ""
        }
    except Exception as e:
        print(f"An error occurred during OpenAI API call for statement '{statement}': {e}")
        return {
            "statement": statement,
            "classification_type": classification_type,
            "is_true": False,
            "reason": f"API call failed: {e}",
            "confidence": 0.0,
            "utility_description": ""
        }

def verify_and_describe_interpretations_with_llm(original_text, interpretations_list, client, model):
    """
    Uses the OpenAI API to:
    1. Detect whether each interpretation statement is true according to the original text.
    2. Describe in natural language how each true statement is useful for its classification type.
    Executes calls in parallel.

    Args:
        original_text (str): The full original text.
        interpretations_list (list[dict]): A list of dictionaries, each with 'statement' and 'type'.
        client: An initialized OpenAI client instance.
        model (str): The OpenAI model to use.

    Returns:
        list[dict]: A list of dictionaries, each containing:
                    - 'statement': original statement
                    - 'classification_type': original type
                    - 'is_true': boolean
                    - 'reason': string
                    - 'confidence': float
                    - 'utility_description': string (describes usefulness for classification)
              Returns an empty list if an API error occurs for all interpretations.
    """
    # print(f"Processing {len(interpretations_list)} interpretations for parallel execution...")
    results = []
    
    # Use ThreadPoolExecutor for parallel API calls (I/O bound)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Submit tasks for each interpretation
        future_to_interpretation = {
            executor.submit(_call_llm_for_single_interpretation, client, model, original_text, item): item
            for item in interpretations_list
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_interpretation):
            interpretation_item = future_to_interpretation[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Interpretation '{interpretation_item.get('statement', 'N/A')}' generated an exception: {exc}")
                results.append({
                    "statement": interpretation_item.get('statement', 'N/A'),
                    "classification_type": interpretation_item.get('type', 'N/A'),
                    "is_true": False,
                    "reason": f"Processing failed: {exc}",
                    "confidence": 0.0,
                    "utility_description": ""
                })
    return results

def _compress_one_class(client, model, class_name, desc_list):
    """
    把同一分类的多句 utility 描述压缩成一句 1 行摘要
    """
    system = (
    "You are a medical text summarizer. "
    f"Given several verified, true statements explaining why a text belongs to '{class_name}', "
    f"extract the unique influence elements and compress them into **one coherent sentence**. "
    f"Do not use bullet points; instead, craft a flowing summary that captures the **combined impact** "
    f"of these elements on the classification."
)
    user = "\n".join(desc_list) + '/no_think'

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            temperature=0.1
        )
        summary = re.sub(r'<think>.*?</think>\n?', '', resp.choices[0].message.content.strip(), flags=re.S).strip()
        return class_name, summary
    except Exception as e:
        return class_name, f"[compression error: {e}]"

def combine_statements_description(true_statements_info, client, model, max_workers=8):
    """
    并发压缩 → 输出一句连贯总结
    """
    if not true_statements_info:
        return "No true statements to combine for a whole description."

    # 1. 按分类聚合
    grouped = defaultdict(set)  # 用 set 自动去重
    for item in true_statements_info:
        grouped[item['classification_type']].add(item['utility_description'])

    if not grouped:
        return "No useful statements found to generate a combined description."

    # 2. 并发压缩
    summaries = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [
            exe.submit(_compress_one_class, client, model, cls, list(descs))
            for cls, descs in grouped.items()
        ]
        for f in as_completed(futures):
            cls, summary = f.result()
            summaries[cls] = summary

    # 3. 拼成一句连贯描述
    parts = [f"{cls}: {summaries[cls]}" for cls in sorted(summaries)]
    combined = "Overall, this text indicates properties relevant for classification, such as " + "; ".join(parts) + "."
    return combined

def _process_single_patient(args):
    text_id, interpretations, original_text, client, model = args
    verified = verify_and_describe_interpretations_with_llm(
        original_text, interpretations, client, model
    )
    if not verified:
        return None

    true_info = [v for v in verified if v.get('is_true')]
    combined = combine_statements_description(true_info, client, model)
    return {
        "text_id": text_id,
        "original_text": original_text,
        "interpretations_details": verified,
        "combined_classification_description": combined or "Failed to generate."
    }
    
# --- Main Execution ---
if __name__ == "__main__":
    import os
    import tqdm
    client = OpenAI(api_key='EMPTY', base_url="http://0.0.0.0:1212/v1")
    # Define your JSONL input file path
    input_jsonl_file = "/home/sevan/myHypotheSAEs/result_cache/concurrent_one_vs_rest/test_interpretations.jsonl"
    profile_file = "com_data/test_profile_concurrent.jsonl"
    OPENAI_MODEL = "Qwen/Qwen3-32B"  # Specify your OpenAI model here
    with open(profile_file, 'r', encoding='utf-8') as f:
        original_data = {json.loads(line.strip())['idx']: json.loads(line.strip())['profile'] for line in f}
    
    # Output file to save the verified results and combined descriptions
    output_jsonl_file = "test_verified_and_combined_interpretations.jsonl"

    # print(f"\nProcessing interpretations from {input_jsonl_file}...")
    interpretation_dict = create_interpretation_classification_dict(input_jsonl_file)

    all_results = []
    
    # --- 主流程替换原来的串行 for ---
    
    with ThreadPoolExecutor(max_workers=8) as outer_exe:
        patient_futures = [
            outer_exe.submit(_process_single_patient, (tid, interps, original_data[tid], client, OPENAI_MODEL))
            for tid, interps in interpretation_dict.items()
        ]

        # 外层 tqdm
        for fut in tqdm.tqdm(as_completed(patient_futures), total=len(patient_futures), desc="Patients"):
            res = fut.result()
            if res:
                all_results.append(res)
                with open(output_jsonl_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(res) + '\n')
    print(f"\nAll interpretations processed. Final results saved to {output_jsonl_file}")
