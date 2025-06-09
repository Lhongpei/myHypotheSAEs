import pandas as pd
import pandas as pd
import pandas as pd

def df_to_prompts(
    few_shot_row,
    few_shot_label,
    target_rows=None,
    output_path=None,
    few_shot_examples=3,
):
    """
    Generate discharge location prediction prompts with few-shot examples.

    Args:
        few_shot_row (pd.DataFrame or List[pd.Series]): Few-shot example rows.
        few_shot_label (List[str]): Few-shot example labels.
        target_rows (pd.DataFrame or List[pd.Series] or pd.Series): Target rows for prediction.
        output_path (str): Output file base path (e.g. 'prompt.txt' or 'prompt.jsonl').
        few_shot_examples (int): Number of examples to include.

    Returns:
        List[str]: List of prompt strings (one per target row).
    """

    # 🔧 Auto-generate field descriptions from few_shot_row
    if isinstance(few_shot_row, pd.DataFrame):
        example_columns = few_shot_row.columns
    elif isinstance(few_shot_row, list):
        example_columns = few_shot_row[0].index
    else:
        raise ValueError("few_shot_row must be DataFrame or list of Series")

    field_descriptions = {
        col: col.replace("_", " ").capitalize() + ": {}"
        for col in example_columns
    }

    # ✅ Normalize few_shot_row to list of Series
    if isinstance(few_shot_row, pd.DataFrame):
        few_shot_row = [r for _, r in few_shot_row.iterrows()]

    # ✅ Normalize target_rows
    if target_rows is None:
        target_rows = []
    elif isinstance(target_rows, pd.DataFrame):
        target_rows = [r for _, r in target_rows.iterrows()]
    elif isinstance(target_rows, pd.Series):
        target_rows = [target_rows]

    # 🔧 Few-shot example block
    few_shot_lines = []
    for i in range(min(few_shot_examples, len(few_shot_row))):
        row = few_shot_row[i]
        label = few_shot_label[i]
        parts = []
        for field, template in field_descriptions.items():
            if field in row and not pd.isna(row[field]) and str(row[field]).lower() != "unknown":
                parts.append(template.format(row[field]))
        few_shot_lines.append(f"Input: {', '.join(parts)}\nOutput: {label}")

    prompt_header = (
        "You are a medical assistant. Based on the patient's personal and medical admission information, "
        "predict the discharge location.\n\nHere are some examples:\n\n"
    )
    few_shot_block = prompt_header + "\n\n".join(few_shot_lines)

    # 🔧 For each target row, build prompt
    prompts = []
    for idx, row in enumerate(target_rows):
        parts = []
        for field, template in field_descriptions.items():
            if field in row and not pd.isna(row[field]) and str(row[field]).lower() != "unknown":
                parts.append(template.format(row[field]))
        prompt = few_shot_block + f"\n\nNow, given a new patient:\nInput: {', '.join(parts)}\nOutput:"
        prompts.append(prompt)

        # Optional output file
        if output_path:
            base, ext = output_path.rsplit(".", 1)
            out_path = f"{base}_{idx}.{ext}" if len(target_rows) > 1 else output_path
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"Saved prompt to {out_path}")

    return prompts



# Example usage
if __name__ == "__main__":
    df_to_prompts(
        csv_path="patients.csv",
        output_path="discharge_prompt.txt"
    )