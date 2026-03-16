# -*- coding: utf-8 -*-
"""12_LLM_surprisal_markers.ipynb

# Prep
"""

import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import os
import math
import csv
import shutil, sys

from google.colab import drive
drive.mount('/content/drive')
result_folder = '/content/drive/todo/'

df = pd.read_csv(result_folder + 'todo.csv', index_col = 0)

df_text = pd.read_csv(result_folder + 'todo.csv', index_col = 0)

def response_surp_mean(response, len, model):
  if len > 1: # need at least two words to have a surp score
    # Sequence Surprisal, normalized by number of tokens - lambda x: -x.mean(0).item()
    score = model.sequence_score([response], reduction = lambda x: -x.mean(0).item())
    return round(score[0],2)
  else:
    return np.nan

!pip install minicons

from minicons import scorer
import torch
import sys

available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(available_gpus)
print(gpu_names)

"""# GPT2 surp"""

gpt2 = scorer.IncrementalLMScorer('gpt2', device)

response_surp_mean(df_text['clean_text'][0], len(df_text['clean_text'][0].split()), gpt2)

df_text['gpt2_surp'] = ''

for i in range(len(df_text)):
  if len(df_text['clean_text'][i].split()) > 1 and i > 180:
    df_text['gpt2_surp'][i] = response_surp_mean(df_text['clean_text'][i], len(df_text['clean_text'][i].split()), gpt2)
    df_text.to_csv(result_folder + 'todo.csv')
    if i % 5 == 0:
      print('finished line: ', i)

df_text['gpt2_surp'] = pd.to_numeric(df_text['gpt2_surp'], errors='coerce')
df_text['gpt2_surp'].describe()

hc_social_text = pd.read_csv(result_folder + 'todo.csv', index_col = 0)
hc_social_text.head()

hc_social_text['gpt2_surp'] = ''

hc_social_text['text'][0] # double check

for i in range(len(hc_social_text)):
    if i > -1 and type(hc_social_text['clean_text'][i]) != float and len(hc_social_text['clean_text'][i].split()) > 1:
      hc_social_text['gpt2_surp'][i] = response_surp_mean(hc_social_text['clean_text'][i], len(hc_social_text['clean_text'][i].split()), gpt2)
      hc_social_text.to_csv(result_folder + 'Health_social_result.csv')
      if i % 5 == 0:
        print('finished line: ', i)

hc_social_text['gpt2_surp'] = pd.to_numeric(hc_social_text['gpt2_surp'], errors='coerce')
hc_social_text['gpt2_surp'].describe()

# Calculate text lengths
df_text['text_length'] = df_text['clean_text'].str.len()
hc_social_text['text_length'] = hc_social_text['clean_text'].str.len()

matched_indices = []
for _, row in df_text.iterrows():
    text_length = row['text_length']
    closest_match = hc_social_text.iloc[(hc_social_text['text_length'] - text_length).abs().argsort()[:1]]
    matched_indices.append(closest_match.index.values[0])

# Subsample from hc_social_text
hc_social_text_subsample = hc_social_text.loc[matched_indices]

# Print descriptive statistics for gpt2_surp
print("Descriptive Statistics for df_text['gpt2_surp']:")
print(df_text['gpt2_surp'].describe())

print("\nDescriptive Statistics for hc_social_text_subsample['gpt2_surp']:")
print(hc_social_text_subsample['gpt2_surp'].describe())

print("Descriptive Statistics for df_text['text_length']:")
print(df_text['text_length'].describe())

print("\nDescriptive Statistics for hc_social_text_subsample['text_length']:")
print(hc_social_text_subsample['text_length'].describe())

"""# BERT"""

bert = scorer.MaskedLMScorer('bert-base-uncased', device)

df_text['bert_surp'] = ''

for i in range(len(df_text)):
  if len(df_text['clean_text'][i].split()) > 1 and i != 180:
    df_text['bert_surp'][i] = response_surp_mean(df_text['clean_text'][i], len(df_text['clean_text'][i].split()), bert)
    df_text.to_csv(result_folder + 'AD_social_result_reduced.csv')
    if i % 5 == 0:
      print('finished line: ', i)

df_text['clean_text'][i]

df_text['gpt2_surp'] = pd.to_numeric(df_text['gpt2_surp'], errors='coerce')
df_text['gpt2_surp'].describe()
