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
from optimizer import OptimizerAE
from model import GCNModelVAE, GCNModelAE
from preprocess import normalize_adj, construct_feed_dict
from utils import visualize_triangular, visualize_matrix, visualize_latent_space

def get_next_batch(batch_size, adj, adj_norm):
    adj_idx = np.random.randint(adj_norm.shape[0], size=batch_size)
    adj_norm_batch = adj_norm[adj_idx, :, :]
    adj_norm_batch = np.reshape(adj_norm_batch, [args.batch_size, num_nodes, num_nodes])
    adj_orig_batch = adj[adj_idx, :, :]
    adj_orig_batch = np.reshape(adj_orig_batch, [args.batch_size, num_nodes, num_nodes])
    return adj_norm_batch, adj_orig_batch, adj_idx
    
# Default settings
class args:
    data_dir = "BSNIP_left_full/"
    hidden_dim_1 = 100
    hidden_dim_2 = 50
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
model = GCNModelAE(placeholders, num_features, num_nodes, args)

# Optimizer
with tf.name_scope('optimizer'):
        opt = OptimizerAE(preds=model.reconstructions,
                           labels=tf.reshape(placeholders['adj_orig'], [-1]),
                           model=model, num_nodes=num_nodes,
                           learning_rate=args.learning_rate)

# Initialize session
sess = tf.Session()

# Train model
print("START TRAINING")
saver = tf.train.Saver()
model_name = "./models/brain_vgae_100_50_autoencoder=True.ckpt"

with sess as sess:
    saver.restore(sess, model_name)

    features_batch = np.zeros([args.batch_size, num_nodes, num_features], dtype=np.float32)
    for i in features_batch:
        np.fill_diagonal(i, 1.)

    adj_norm, adj_orig, adj_idx = get_next_batch(args.batch_size, adj, adj_norm)
    features = features_batch
    feed_dict = construct_feed_dict(adj_norm, adj_orig, features, placeholders)
    feed_dict.update({placeholders['dropout']: args.dropout})
    outs = sess.run([model.reconstructions, model.z_mean, opt.cost], feed_dict=feed_dict)
    reconstructions = outs[0].reshape([args.batch_size, 180, 180])
    z_mean = outs[1]
    cost = outs[2]
    
    # Visualize sample full matrix of original, normalized, and reconstructed batches. 
    print('original: ', adj_orig[1])
    print('reconstruction: ', reconstructions[1])
    import pdb; pdb.set_trace()
    visualize_matrix(adj_orig, 1, 'original')
    visualize_matrix(adj_norm, 1, 'normalized')
    visualize_matrix(reconstructions, 1, 'reconstruction')
    
    # Visualize Latent Space
    onehot = np.array([0 if idx < 203 else 1 for idx in adj_idx])
    visualize_latent_space(z_mean, onehot)
