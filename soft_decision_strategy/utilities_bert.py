"""
This script has utilities for the BERT model.
"""


import numpy as np

def proba_to_labels(proba):
  labels = np.zeros_like(proba)
  max_indices = np.argmax(proba, axis=1)
  for i in range(len(proba)):
    labels[i][max_indices[i]] = 1
  return labels