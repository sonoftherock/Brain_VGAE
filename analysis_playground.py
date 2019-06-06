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
from model import GCNModelVAE, GCNModelAE
from preprocess import normalize_adj, construct_feed_dict
from utils import visualize_triangular, visualize_matrix, visualize_latent_space, get_random_batch, get_consecutive_batch

# Default settings
class args:
    data_dir = "BSNIP_left_full/"
    hidden_dim_1 = 100
    hidden_dim_2 = 50
    hidden_dim_3 = 5
    batch_size = 32
    learning_rate = 0.0001
    dropout = 0.

# Load data
adj = load_data("./data/" + args.data_dir + "ignore_negative.npy")

for sub in adj:
    np.fill_diagonal(sub, 1)

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
'dropout': tf.placeholder_with_default(tf.cast(0., tf.float32), shape=())
}

# Create model
model = GCNModelVAE(placeholders, num_features, num_nodes, args)

# Initialize session
sess = tf.Session()

# Train model
saver = tf.train.Saver()
model_name = './models/brain_vgae_ignore_negative_100_50_5_autoencoder=False.ckpt'
print("Analyzing " + model_name)

with tf.Session() as sess:
    saver.restore(sess, model_name)

    features_batch = np.zeros([args.batch_size, num_nodes, num_features], dtype=np.float32)
    for i in features_batch:
        np.fill_diagonal(i, 1.)

    adj_norm_batch, adj_orig_batch, adj_idx = get_consecutive_batch(0, args.batch_size, adj, adj_norm)
    features = features_batch
    feed_dict = construct_feed_dict(adj_norm_batch, adj_orig_batch, features, placeholders)
    feed_dict.update({placeholders['dropout']: args.dropout})
    outs = sess.run([model.reconstructions, model.z_mean], feed_dict=feed_dict)
    
    reconstructions = outs[0].reshape([args.batch_size, 180, 180])
    z_mean = outs[1]
#     Visualize sample full matrix of original, normalized, and reconstructed batches. 
    for i in range(adj_orig_batch.shape[0]):
        visualize_matrix(adj_orig_batch, i, model_name, 'original_' + str(i))
        visualize_matrix(adj_norm_batch, i, model_name, 'normalized_' + str(i))
        visualize_matrix(reconstructions, i, model_name, 'reconstruction_' + str(i))
        
    idx_all, z_all = [], []
    for i in range(10):
        adj_norm_batch, adj_orig_batch, adj_idx = get_random_batch(args.batch_size, adj, adj_norm)
        features = features_batch
        feed_dict = construct_feed_dict(adj_norm_batch, adj_orig_batch, features, placeholders)
        feed_dict.update({placeholders['dropout']: args.dropout})
        outs = sess.run([model.reconstructions, model.z_mean], feed_dict=feed_dict)
        idx_all.append(adj_idx)
        z_all.append(outs[1])
    
#     Visualize Latent Space
    z = np.array(z_all).reshape(-1, 10)
    idx = np.array(idx_all).flatten()
    onehot = np.array([0 if i < 203 else 1 for i in idx_all[0]])
    visualize_latent_space(z_all[0], onehot, model_name)
