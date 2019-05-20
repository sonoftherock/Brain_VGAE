import tensorflow as tf


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, learning_rate):
        self.preds_sub = preds
        self.labels_sub = labels
        # self.cost = tf.reduce_mean(self.labels_sub)
        self.cost = tf.reduce_mean(tf.square(self.labels_sub - self.preds_sub))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(tf.clip_by_value(model.z_log_std, -1., 1.))), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
