import tensorflow as tf


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, learning_rate, kl_coefficient):
        self.preds_sub = preds
        self.labels_sub = labels
#         self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.rc_loss = tf.reduce_mean(tf.square(self.labels_sub - self.preds_sub))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer

        # Latent loss
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost = self.rc_loss - kl_coefficient * self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

class OptimizerAE(object):
    def __init__(self, preds, labels, model, num_nodes, learning_rate):
        self.preds_sub = preds
        self.labels_sub = labels
        self.rc_loss = tf.reduce_mean(tf.square(self.labels_sub - self.preds_sub))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # Adam Optimizer

        # Latent loss
        self.cost = self.rc_loss 
        
        # Just to let train script run, doesn't do anything.
        self.kl = tf.zeros([1,1])

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
