import numpy as np
import tensorflow as tf

def normalize_adj_tf(adj):
    rowsum = np.array(tf.reduce_sum(adj,2))
    adj_norm = np.zeros(adj.shape, np.float32)
    for i in range(rowsum.shape[0]):
        degree_mat_inv_sqrt = np.diag(np.sign(rowsum[i])*np.power(np.abs(rowsum[i]), -0.5).flatten())
        degree_mat_inv_sqrt[np.isinf(degree_mat_inv_sqrt)] = 0.
        adj_norm[i] = adj[i].dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_norm

def normalize_adj(adj):
    rowsum = np.array(adj.sum(2))
    adj_norm = np.zeros(adj.shape, np.float32)
    for i in range(rowsum.shape[0]):
        degree_mat_inv_sqrt = np.diag(np.sign(rowsum[i])*np.power(np.abs(rowsum[i]), -0.5).flatten())
        degree_mat_inv_sqrt[np.isinf(degree_mat_inv_sqrt)] = 0.
        adj_norm[i] = adj[i].dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_norm



def construct_feed_dict(adj_norm, adj_orig, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj_norm']: adj_norm})
    feed_dict.update({placeholders['adj_orig']: adj_orig})
    return feed_dict

def preprocess_graph(data_dir):
    dataset = np.load('./data/' + data_dir + '/original.npy')
    np.save('./data/' + data_dir + '/features.npy', dataset)
    threshold = np.mean(dataset)
    for subject in dataset:
        subject[subject > threshold] = 1
        subject[subject <= threshold] = 0

        # Make sure every node is connected to itself.
        np.fill_diagonal(subject, 1)

    np.save('./data/' + data_dir + '/adj.npy', dataset)
preprocess_graph('BSNIP_left_full')
