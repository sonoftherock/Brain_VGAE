import numpy as np
import sys

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(adj_file, features_file):
    adj = np.load(adj_file)
    features = np.load(features_file)
    return adj, features
