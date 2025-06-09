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

base_dir = os.path.join(prefix, "demo-data")
train_df = pd.read_json(os.path.join(base_dir, "yelp-demo-train-20K.json"), lines=True)
val_df = pd.read_json(os.path.join(base_dir, "yelp-demo-val-2K.json"), lines=True)

texts = train_df['text'].tolist()
labels = train_df['stars'].values
val_texts = val_df['text'].tolist() # These are only used for early stopping of SAE training, so we don't need labels.

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

TASK_SPECIFIC_INSTRUCTIONS = """All of the texts are reviews of restaurants on Yelp.
Features should describe a specific aspect of the review. For example:
- "mentions long wait times to receive service"
- "praises how a dish was cooked, with phrases like 'perfect medium-rare'\""""
print(texts[0])  # Print an example text to see the format
print(train_embeddings[0])  # Print the corresponding embedding to see the format
print(sae_list[0])
# Interpret random neurons
results = interpret_sae(
    texts=texts,
    embeddings=train_embeddings,
    sae=sae_list,
    n_random_neurons=5,
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
    task_specific_instructions=TASK_SPECIFIC_INSTRUCTIONS
)

print("\nMost predictive features of Yelp reviews:")
pd.set_option('display.max_colwidth', None)
display(results.sort_values(by=f"target_{selection_method}", ascending=False))
pd.reset_option('display.max_colwidth')

