import time
import os
import argparse

# Using Tesla K80 on Yale CS cluster (tangra)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import numpy as np

from optimizer import OptimizerVAE
from input_data import load_data
from model import GCNModelVAE
from preprocess import normalize_adj, construct_feed_dict

# Settings
parser = argparse.ArgumentParser()
parser.parse_args()
parser.add_argument("data_dir", nargs='?', type=str, default="BSNIP_left_full/")
parser.add_argument("learning_rate", nargs='?', type=float, default=0.001)
parser.add_argument("epochs", nargs='?', type=int, default=10000)
parser.add_argument("batch_size", nargs='?', type=int, default=32)
parser.add_argument("hidden_dim_1", nargs='?', type=int, default=200)
parser.add_argument("hidden_dim_2", nargs='?', type=int, default=100)
parser.add_argument("dropout", nargs='?', type=float, default=0.)
args = parser.parse_args()
print("Learning Rate: " + str(args.learning_rate))
print("Hidden dimensions: " + str(args.hidden_dim_1), " ", str(args.hidden_dim_2))

# Load data
adj, features = load_data("./data/" + args.data_dir + "adj.npy", "./data/" + args.data_dir + "features.npy")

# Normalize adjacency matrix (i.e. D^(.5)AD^(.5))
adj_norm = normalize_adj(adj)

num_nodes = adj.shape[1]
num_features = features.shape[1]

# Define placeholders
placeholders = {
'features': tf.placeholder(tf.float32, [args.batch_size, num_nodes, num_features]),
'adj_norm': tf.placeholder(tf.float32, [args.batch_size, num_nodes, num_nodes]),
'adj_orig': tf.placeholder(tf.float32, [args.batch_size, num_nodes, num_nodes]),
'dropout': tf.placeholder_with_default(0., shape=())
}

# Create model
model = GCNModelVAE(placeholders, num_features, num_nodes, args)

# pos_weight = float(adj.shape[1] * adj.shape[1] - adj.sum()) / adj.sum()
# norm = adj.shape[1] * adj.shape[1] / float((adj.shape[1] * adj.shape[1] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    opt = OptimizerVAE(preds=model.reconstructions,
                       labels=tf.reshape(placeholders['adj_orig'], [-1]),
                       model=model, num_nodes=num_nodes,
                       learning_rate=args.learning_rate)

def get_next_batch(batch_size, adj, adj_norm):
    adj_idx = np.random.randint(adj_norm.shape[0], size=batch_size)
    adj_norm_batch = adj_norm[adj_idx, :, :]
    adj_norm_batch = np.reshape(adj_norm_batch, [batch_size, num_nodes, num_nodes])
    adj_orig_batch = adj[adj_idx, :, :]
    adj_orig_batch = np.reshape(adj_orig_batch, [batch_size, num_nodes, num_nodes])
    features_batch = features[adj_idx,:,:]
    features_batch = np.reshape(features_batch, [batch_size, num_nodes, num_features])
    return adj_norm_batch, adj_orig_batch, features_batch

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train model
print("START TRAINING")
for epoch in range(args.epochs):
    t = time.time()
    adj_norm, adj_orig, features = get_next_batch(args.batch_size, adj, adj_norm)
    feed_dict = construct_feed_dict(adj_norm, adj_orig, features, placeholders)
    feed_dict.update({placeholders['dropout']: args.dropout})
    outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    # if epoch % 100 == 0:
    print(outs)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")