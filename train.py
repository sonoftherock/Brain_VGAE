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
from tensorflow.python import debug as tf_debug

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", nargs='?', help="data directory", type=str, default="BSNIP_left_full/")
parser.add_argument("learning_rate", nargs='?', type=float, default=0.0001)
parser.add_argument("epochs", nargs='?', type=int, default=10000)
parser.add_argument("batch_size", nargs='?', type=int, default=32)
parser.add_argument("hidden_dim_1", nargs='?', type=int, default=200)
parser.add_argument("hidden_dim_2", nargs='?', type=int, default=100)
parser.add_argument("dropout", nargs='?', type=float, default=0.)
parser.add_argument('--debug', help='turn on tf debugger', action="store_true")

args = parser.parse_args()
print("Learning Rate: " + str(args.learning_rate))
print("Hidden dimensions: " + str(args.hidden_dim_1), " ", str(args.hidden_dim_2))

# Load data
# adj, features = load_data("./data/" + args.data_dir + "adj.npy", "./data/" + args.data_dir + "features.npy")
adj = load_data("./data/" + args.data_dir + "original.npy")
print('LOADING DATA FROM: ' + "./data/" + args.data_dir + "original.npy")

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
model = GCNModelVAE(placeholders, num_features, num_nodes, args)

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
    # features_batch = features[adj_idx,:,:]
    # features_batch = np.reshape(features_batch, [batch_size, num_nodes, num_features])
    return adj_norm_batch, adj_orig_batch

# Initialize session
sess = tf.Session()

if args.debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

# Train model
print("START TRAINING")
saver = tf.train.Saver()
model = "../models/brain_vgae_200_100.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, model)
    start_time = time.time()
    features_batch = np.zeros([args.batch_size, num_nodes, num_features], dtype=np.float32)
    for i in features_batch:
        np.fill_diagonal(i, 1.)

    for epoch in range(args.epochs):
        t = time.time()
        adj_norm, adj_orig = get_next_batch(args.batch_size, adj, adj_norm)
        features = features_batch
        feed_dict = construct_feed_dict(adj_norm, adj_orig, features, placeholders)
        feed_dict.update({placeholders['dropout']: args.dropout})
        input, reconstruction = sess.run([opt.labels_sub , opt.preds_sub], feed_dict=feed_dict)
        outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "time=", "{:.5f}".format(time.time() - t))
    save_path = saver.save(sess, model)
    print('done saving at',save_path)
