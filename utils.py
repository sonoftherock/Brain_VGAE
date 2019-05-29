import matplotlib.pyplot as plt
import matplotlib.colors as clr
plt.switch_backend('agg')
import tensorflow as tf
import numpy as np

def visualize_triangular(batch, idx):
    tri = np.zeros((180, 180))
    tri[np.triu_indices(180,1)] = batch[idx]
    plt.imshow(tri, vmin=-1, vmax=1, cmap="RdBu")
    plt.show()
    plt.clf()
    
def visualize_matrix(batch, idx, model_name, name):
    tri = batch[idx].reshape((180,180))
    plt.imshow(tri, vmin=-1, vmax=1, cmap="RdBu")
    plt.colorbar()
    plt.savefig("./plots/" + model_name[9:] + "/" + name)
    plt.clf()

def visualize_latent_space(z_mean, labels, model_name):
    plt.figure(figsize=(14,14))
    # Check first 5 nodes
    for k in range(5):
        for i in range(5):
            for j in range(5):
                plt.subplot(5,5,i+1+j*5)
                plt.plot(z_mean[labels==0,k,i], z_mean[labels==0,k,j], 'o', label='Control')
                plt.plot(z_mean[labels==1,k,i], z_mean[labels==1,k,j], 'o', label='Schizophrenic')
        plt.legend()        
        plt.savefig('./plots/' + model_name[9:] + "/" + 'latent_space_%i.png' %(k))
        plt.clf()
    
def get_random_batch(batch_size, adj, adj_norm):
    adj_idx = np.random.randint(adj.shape[0], size=batch_size)
    num_nodes = adj.shape[1]
    adj_norm_batch = adj_norm[adj_idx, :, :]
    adj_norm_batch = np.reshape(adj_norm_batch, [batch_size, num_nodes, num_nodes])
    adj_orig_batch = adj[adj_idx, :, :]
    adj_orig_batch = np.reshape(adj_orig_batch, [batch_size, num_nodes, num_nodes])
    return adj_norm_batch, adj_orig_batch, adj_idx

def get_consecutive_batch(start, batch_size, adj, adj_norm):
    adj_idx = np.arange(start, start + batch_size)
    num_nodes = adj.shape[1]
    adj_norm_batch = adj_norm[adj_idx, :, :]
    adj_norm_batch = np.reshape(adj_norm_batch, [batch_size, num_nodes, num_nodes])
    adj_orig_batch = adj[adj_idx, :, :]
    adj_orig_batch = np.reshape(adj_orig_batch, [batch_size, num_nodes, num_nodes])
    return adj_norm_batch, adj_orig_batch, adj_idx
