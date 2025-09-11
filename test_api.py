# HypotheSAEs Quickstart

import os
os.environ['OPENAI_KEY_SAE'] = 'EMPTY' # Replace with your OpenAI API key, or with another environment variable (e.g. os.environ['OPENAI_API_
import numpy as np
import pandas as pd

from hypothesaes.quickstart import train_sae, interpret_sae, generate_hypotheses, evaluate_hypotheses
from hypothesaes.embedding import get_openai_embeddings, get_local_embeddings