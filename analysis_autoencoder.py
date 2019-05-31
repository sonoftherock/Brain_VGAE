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
from optimizer import OptimizerAE, OptimizerVAE
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
    kl_coefficient = 0.0001
    activation='tanh'
    dropout = 0.

# Load data
adj = load_data("./data/" + args.data_dir + "original.npy")

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
'dropout': tf.placeholder_with_default(0., shape=())
}

# Create model
model = GCNModelAE(placeholders, num_features, num_nodes, args)

# Initialize session
sess = tf.Session()

# Train model
saver = tf.train.Saver()
# model_name = "./models/brain_vgae_100_50_autoencoder=False_kl_coefficient=0.001_act=tanh.ckpt"
model_name = "./models/brain_vgae_100_50_autoencoder=True.ckpt"

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
    outs = sess.run([model.reconstructions], feed_dict=feed_dict)
    
    reconstructions = outs[0].reshape([args.batch_size, 180, 180])
    
#     Visualize sample full matrix of original, normalized, and reconstructed batches. 
    for i in range(adj_orig_batch.shape[0]):
        visualize_matrix(adj_orig_batch, i, model_name, 'original_' + str(i))
        visualize_matrix(adj_norm_batch, i, model_name, 'normalized_' + str(i))
        visualize_matrix(reconstructions, i, model_name, 'reconstruction_' + str(i))

    adj_norm_batch, adj_orig_batch, adj_idx = get_random_batch(args.batch_size, adj, adj_norm)
    features = features_batch
    feed_dict = construct_feed_dict(adj_norm_batch, adj_orig_batch, features, placeholders)
    feed_dict.update({placeholders['dropout']: args.dropout})
    outs = sess.run([model.z_mean], feed_dict=feed_dict)
    
    z = outs[0]
    
    # Visualize Latent Space
    onehot = np.array([0 if idx < 203 else 1 for idx in adj_idx])
    visualize_latent_space(z, onehot, model_name)
