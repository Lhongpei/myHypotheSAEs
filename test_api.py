# HypotheSAEs Quickstart
# This notebook demonstrates basic usage of HypotheSAEs on a sample of the Yelp review dataset

# %load_ext autoreload
# %autoreload 2

import os
os.environ['OPENAI_KEY_SAE'] = 'EMPTY' # Replace with your OpenAI API key, or with another environment variable (e.g. os.environ['OPENAI_API_
import numpy as np
import pandas as pd

from hypothesaes.quickstart import train_sae, interpret_sae, generate_hypotheses, evaluate_hypotheses
from hypothesaes.embedding import get_openai_embeddings, get_local_embeddings

current_dir = os.getcwd()
if current_dir.endswith("notebooks"):
    prefix = "../"
else:
    prefix = "./"
val_ratio = 0.1  # Ratio of training data to use for validation
import sklearn
import sklearn.model_selection
few_shot_examples = 5
# base_dir = os.path.join(prefix, "demo-data")
# train_df = pd.read_json(os.path.join(base_dir, "yelp-demo-train-20K.json"), lines=True)
# val_df = pd.read_json(os.path.join(base_dir, "yelp-demo-val-2K.json"), lines=True)

# texts = train_df['text'].tolist()
# labels = train_df['stars'].values
# val_texts = val_df['text'].tolist() # These are only used for early stopping of SAE training, so we don't need labels.
from utils import df_to_prompts
base_dir = os.path.join(prefix, 'data')
train_X = pd.read_csv(os.path.join(base_dir, "X_train.csv"))
train_y = pd.read_csv(os.path.join(base_dir, "y_train.csv")).values.ravel()
test_X = pd.read_csv(os.path.join(base_dir, "X_test.csv"))
test_y = pd.read_csv(os.path.join(base_dir, "y_test.csv")).values.ravel()
number_dict = {'home': 0, 'other': 0, 'died': 1}
label_train = [number_dict[label] for label in train_y]
label_test = [number_dict[label] for label in test_y]
few_shot_row = train_X.iloc[0:few_shot_examples, :]
few_shot_label = label_train[0:few_shot_examples]
train_texts = df_to_prompts(few_shot_row, few_shot_label, train_X.iloc[few_shot_examples:, :], few_shot_examples=few_shot_examples)
texts, val_texts, labels, val_labels = sklearn.model_selection.train_test_split(
    train_texts, label_train[few_shot_examples:], test_size=val_ratio, random_state=42, shuffle=True
)
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B" # OpenAI
# EMBEDDER = "nomic-ai/modernbert-embed-base" # Huggingface model, will run locally
CACHE_NAME = f"yelp_quickstart_{EMBEDDER}"

# text2embedding = get_openai_embeddings(texts + val_texts, model=EMBEDDER, cache_name=CACHE_NAME)
text2embedding = get_local_embeddings(texts + val_texts, model=EMBEDDER, batch_size=128, cache_name=CACHE_NAME)
embeddings = np.stack([text2embedding[text] for text in texts])

train_embeddings = np.stack([text2embedding[text] for text in texts])
val_embeddings = np.stack([text2embedding[text] for text in val_texts])

checkpoint_dir = os.path.join(prefix, "checkpoints", CACHE_NAME)
sae_256_8 = train_sae(embeddings=train_embeddings, M=256, K=8, checkpoint_dir=checkpoint_dir, val_embeddings=val_embeddings)
sae_32_4 = train_sae(embeddings=train_embeddings, M=32, K=4, checkpoint_dir=checkpoint_dir, val_embeddings=val_embeddings)
sae_list = [sae_256_8, sae_32_4]

# This instruction will be included in the neuron interpretation prompt.
# The below instructions are specific to Yelp, but you can customize this for your task.
# If you don't pass in task-specific instructions, there is a generic instruction (see src/interpret_neurons.py);
# task-specific instructions are optional, but they help produce hypotheses at the desired level of specificity.

# This instruction will be included in the neuron interpretation prompt.
# The below instructions are specific to Yelp, but you can customize this for your task.
# If you don't pass in task-specific instructions, there is a generic instruction (see src/interpret_neurons.py);
# task-specific instructions are optional, but they help produce hypotheses at the desired level of specificity.

TASK_SPECIFIC_INSTRUCTIONS = None
# """All of the texts are reviews of restaurants on Yelp.
# Features should describe a specific aspect of the review. For example:
# - "mentions long wait times to receive service"
# - "praises how a dish was cooked, with phrases like 'perfect medium-rare'\""""
# Interpret random neurons
results = interpret_sae(
    texts=texts,
    embeddings=train_embeddings,
    sae=sae_list,
    n_random_neurons=100,
    print_examples_n=3,
    task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS
)

selection_method = "correlation"
results = generate_hypotheses(
    texts=texts,
    labels=labels,
    embeddings=embeddings,
    sae=sae_list,
    cache_name=CACHE_NAME,
    selection_method=selection_method,
    n_selected_neurons=20,
    n_candidate_interpretations=1,
    task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS,
    classification=True
)

# print("\nMost predictive features of Yelp reviews:")
# pd.set_option('display.max_colwidth', None)
# display(results.sort_values(by=f"target_{selection_method}", ascending=False))
# pd.reset_option('display.max_colwidth')
# print("\nMost predictive features of Yelp reviews:")
# pd.set_option('display.max_colwidth', None)
# display(results.sort_values(by=f"target_{selection_method}", ascending=False))
# pd.reset_option('display.max_colwidth')

holdout_texts = df_to_prompts(few_shot_row, few_shot_label, test_X, few_shot_examples=few_shot_examples)
holdout_labels = label_test  # Exclude few-shot examples

metrics, evaluation_df = evaluate_hypotheses(
    hypotheses_df=results,
    texts=holdout_texts,
    labels=holdout_labels,
    cache_name=CACHE_NAME,
)

pd.set_option('display.max_colwidth', None)
# display(evaluation_df)
pd.reset_option('display.max_colwidth')

print("\nHoldout Set Metrics:")
print(f"R² Score: {metrics['r2']:.3f}")
print(f"Significant hypotheses: {metrics['Significant'][0]}/{metrics['Significant'][1]} " 
      f"(p < {metrics['Significant'][2]:.3e})")
