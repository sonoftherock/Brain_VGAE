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
    
def visualize_matrix(batch, idx, name):
    tri = batch[idx].reshape((180,180))
    plt.imshow(tri, vmin=-1, vmax=1, cmap="RdBu")
    plt.colorbar()
    plt.savefig("./plots/" + name)
    plt.clf()

def visualize_latent_space(z_mean, labels):
    plt.figure(figsize=(14,14))
    for i in range(5):
        for j in range(5):
            plt.subplot(5,5,i+1+j*5)
            plt.plot(z_mean[labels==0,0,i], z_mean[labels==0,0,j], 'o', label='Control')
            plt.plot(z_mean[labels==1,0,i], z_mean[labels==1,0,j], 'o', label='Schizophrenic')
    plt.legend()        
    plt.savefig('./plots/latent_space.png')
    plt.show()
    