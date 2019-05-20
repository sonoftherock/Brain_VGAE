import tensorflow as tf


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, learning_rate):
        preds_sub = preds
        labels_sub = labels
        import pdb; pdb.set_trace()
        self.cost = tf.reduce_mean(tf.square(labels_sub - preds_sub))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)

        # Latent loss
        self.log_lik = self.cost
        # self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
        #                                                            tf.square(tf.exp(model.z_log_std)), 1))
        # import pdb; pdb.set_trace()
        # self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
