import matplotlib.pyplot as plt
import matplotlib.colors as clr
plt.switch_backend('agg')
import tensorflow as tf
import numpy as np
import scipy.stats as sp
import math
import time
import argparse
import logging
from input_data import load_data
from optimizer import OptimizerAE, OptimizerVAE
from model import GCNModelVAE, GCNModelAE
from preprocess import normalize_adj, construct_feed_dict
from utils import visualize_triangular, visualize_matrix, visualize_latent_space, get_consecutive_batch, get_random_batch

# Default settings
class args:
    data_dir = "BSNIP_left_full/"
    hidden_dim_1 = 100
    hidden_dim_2 = 50
    hidden_dim_3 = 5
    batch_size = 32
    learning_rate = 0.0001
    dropout = 0.
    kl_coefficient=0.001
    autoencoder = False

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

# Score model
saver = tf.train.Saver()
model_name = './models/brain_vgae_ignore_negative_100_50_5_autoencoder=False.ckpt'

with tf.Session() as sess:
    saver.restore(sess, model_name)
    features_batch = np.zeros([args.batch_size, num_nodes, num_features], dtype=np.float32)
    for i in features_batch:
        np.fill_diagonal(i, 1.)
    
    start, tot_rc_loss = 0, 0
    og, gen_all = [], [] 
    # Get average reconstruction loss on first 353 subjects
    while start + 32 < adj.shape[0]:
        adj_norm_batch, adj_orig_batch, adj_idx = get_consecutive_batch(start, args.batch_size, adj, adj_norm)
        og.append(adj_orig_batch)
        features = features_batch
        feed_dict = construct_feed_dict(adj_norm_batch, adj_orig_batch, features, placeholders)
        feed_dict.update({placeholders['dropout']: args.dropout})
        outs = sess.run([model.reconstructions, model.z_mean], feed_dict=feed_dict)
        
        reconstructions = outs[0].reshape([args.batch_size, 180, 180])
        tot_rc_loss += tf.reduce_mean(tf.square(adj_orig_batch - reconstructions))
        start += args.batch_size
    avg_rc_loss = tot_rc_loss / math.floor(adj.shape[0]/args.batch_size)
    f = open('./scores/%s.txt' % (model_name[9:]), 'w')
    f.write("average reconstruction loss: %f" % avg_rc_loss.eval())
    
    og = np.array(og).reshape(11, 32, -1)
    og = og.reshape(-1, 32400)
    og_mean = np.mean(og, axis=0)
    og_var = np.var(og, axis=0)

    # TODO: Get pearson coefficients of first and second moments (Only for variational models) - make sure latent space is N(0,0.1)?
    for i in range(11):
        randoms = [np.random.normal(0, 1.0, (num_nodes, args.hidden_dim_3)) for _ in range(args.batch_size)]
        [gen] = sess.run([model.reconstructions], feed_dict={model.z: randoms})
        gen = gen.reshape(args.batch_size, -1)
        gen_all.append(gen)
    gen = np.array(gen_all).reshape(11, 32, -1)
    gen = gen.reshape(-1, 32400)
    gen_mean = np.mean(gen, axis=0)
    gen_var = np.var(gen, axis=0)
    print("mean_gen shape: ", gen_mean.shape)
    plt.scatter(og_mean, gen_mean)
    plt.title('Feature Mean')
    plt.xlabel('original')
    plt.ylabel('generated')
    plt.savefig('./scores/%s_mean.png'%(model_name[9:]))
    plt.clf()
    plt.scatter(og_var, gen_var)
    plt.title('Feature Variance')
    plt.xlabel('original')
    plt.ylabel('generated')
    plt.savefig('./scores/%s_variance.png'%(model_name[9:]))
    plt.clf()
    f
    f.write("Feature mean (180 * 180 features): " + str(sp.pearsonr(og_mean, gen_mean)))
    f.write("Feature Variance (180*180 features): " + str(sp.pearsonr(og_var, gen_var)))

    f.close()
