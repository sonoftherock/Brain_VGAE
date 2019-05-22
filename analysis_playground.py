import matplotlib.pyplot as plt
import matplotlib.colors as clr
plt.switch_backend('agg')
import tensorflow as tf
import numpy as np
import scipy.stats as sp
import time
import argparse
import logging
from input_data import load_data
from model import GCNModelVAE
from preprocess import normalize_adj, construct_feed_dict

def get_next_batch(batch_size, adj, adj_norm):
    adj_idx = np.random.randint(adj_norm.shape[0], size=batch_size)
    adj_norm_batch = adj_norm[adj_idx, :, :]
    adj_norm_batch = np.reshape(adj_norm_batch, [args.batch_size, num_nodes, num_nodes])
    adj_orig_batch = adj[adj_idx, :, :]
    adj_orig_batch = np.reshape(adj_orig_batch, [args.batch_size, num_nodes, num_nodes])
    return adj_norm_batch, adj_orig_batch

def visualize(batch, idx):
    tri = np.zeros((180, 180))
    tri[np.triu_indices(180,1)] = batch[idx]
    plt.imshow(tri, vmin=-1, vmax=1, cmap="RdBu")
    plt.show()
    plt.clf()
    
def visualize_full(batch, idx, name):
    tri = batch[idx].reshape((180,180))
    plt.imshow(tri, vmin=-1, vmax=1, cmap="RdBu")
    plt.colorbar()
    plt.savefig("./plots/" + name)
    plt.clf()
    
# Default settings
class args:
    data_dir = "BSNIP_left_full/"
    hidden_dim_1 = 200
    hidden_dim_2 = 100
    batch_size = 32
    learning_rate = 0.0001
    dropout = 0.

# Load data
adj = load_data("./data/" + args.data_dir + "original.npy")

# Normalize adjacency matrix (i.e. D^(.5)AD^(.5))
adj_norm = normalize_adj(adj)

num_nodes = adj.shape[1]

# CHANGE TO features.shape[1] LATER
num_features = adj.shape[1]

    
# Define placeholders
placeholders = {
'features': tf.placeholder(tf.float32, [args.batch_size, num_nodes, num_features]),
'adj_norm': tf.placeholder(tf.float32, [args.batch_size, num_nodes, num_nodes]),
'adj_orig': tf.placeholder(tf.float32, [args.batch_size, num_nodes, num_nodes]),
'dropout': tf.placeholder_with_default(0., shape=())
}

# Create model
model_placeholder = GCNModelVAE(placeholders, num_features, num_nodes, args)


# Initialize session
sess = tf.Session()

# Train model
print("START TRAINING")
saver = tf.train.Saver()
model = "../models/brain_vgae_200_100.ckpt"

with sess as sess:
    saver.restore(sess, model)

    features_batch = np.zeros([args.batch_size, num_nodes, num_features], dtype=np.float32)
    for i in features_batch:
        np.fill_diagonal(i, 1.)

    adj_norm, adj_orig  = get_next_batch(args.batch_size, adj, adj_norm)
    features = features_batch
    feed_dict = construct_feed_dict(adj_norm, adj_orig, features, placeholders)
    feed_dict.update({placeholders['dropout']: args.dropout})
    outs = sess.run([model_placeholder.reconstructions], feed_dict=feed_dict)
    reconstructions = outs[0].reshape([args.batch_size, 180, 180])
    visualize_full(adj_orig, 1, 'original')
    visualize_full(adj_norm, 1, 'normalized')
    visualize_full(reconstructions, 1, 'reconstruction')