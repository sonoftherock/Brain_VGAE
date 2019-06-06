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
    # Account for negative degree matrices
    adj = np.abs(adj)
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

def fill_diagonal(data_dir):
    dataset = np.load('./data/' + data_dir + '/original.npy')

    # Make sure every node is connected to itself.
    for subject in dataset:
        np.fill_diagonal(subject, 1)
    np.save('./data/' + data_dir + '/original.npy', dataset)

def ignore_negative(data_dir):
    dataset = np.load('./data/' + data_dir + '/original.npy')
    dataset = np.clip(dataset, 0, 1)

    # Make sure every node is connected to itself.
    for subject in dataset:
        np.fill_diagonal(subject, 1)
    np.save('./data/' + data_dir + '/ignore_negative.npy', dataset)

def add_min(data_dir):
    dataset = np.load('./data/' + data_dir + '/original.npy')
    threshold = np.amin(dataset)
    dataset = dataset + np.abs(threshold)

    # Make sure every node is connected to itself.
    for subject in dataset:
        np.fill_diagonal(subject, 1 + np.abs(threshold))
    np.save('./data/' + data_dir + '/add_min_adj.npy', dataset)
ignore_negative('BSNIP_left_full')
