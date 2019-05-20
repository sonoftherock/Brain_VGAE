import numpy as np
import sys

# def load_data(adj_file, features_file):
#     adj = np.load(adj_file)
#     features = np.load(features_file)
#     return adj, features

def load_data(dataset):
    return np.load(dataset)
