
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
number_dict = {'home': 0, 'other': 1, 'died': 2}
label_train = [number_dict[label] for label in train_y]
label_test = [number_dict[label] for label in test_y]
few_shot_row = train_X.iloc[0:5, :]
few_shot_label = label_train[0:5]
train_texts = df_to_prompts(few_shot_row, few_shot_label, train_X, few_shot_examples=5)
len(train_texts)