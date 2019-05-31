from layers import GraphConvolution, InnerProductDecoder
import tensorflow as tf
import math
from preprocess import normalize_adj_tf

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def build(self, args):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build(args)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, args, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.batch_size = args.batch_size
        self.input_dim = num_features
        self.n_samples = num_nodes
        self.adj = placeholders['adj_orig']
        self.dropout = placeholders['dropout']
        self.build(args)

    def _build(self, args):
        self.hidden1 = GraphConvolution(batch_size=self.batch_size,
                                              input_dim=self.input_dim,
                                              output_dim=args.hidden_dim_1,
                                              adj=self.adj,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(batch_size=self.batch_size,
                                       input_dim=args.hidden_dim_1,
                                       output_dim=args.hidden_dim_2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.reconstructions = InnerProductDecoder(input_dim=args.hidden_dim_2,
                                      act= tf.nn.tanh,
                                      logging=self.logging)(self.z_mean)

class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, args, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.batch_size = args.batch_size
        self.input_dim = num_features
        self.n_samples = num_nodes
        self.adj = placeholders['adj_orig']
        self.dropout = placeholders['dropout']
        self.build(args)

    def _build(self, args):
        self.hidden1 = GraphConvolution(batch_size=self.batch_size,
                                              input_dim=self.input_dim,
                                              output_dim=args.hidden_dim_1,
                                              adj=self.adj,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)
        
        self.z_mean = GraphConvolution(batch_size=self.batch_size,
                                       input_dim=args.hidden_dim_1,
                                       output_dim=args.hidden_dim_2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(batch_size=self.batch_size,
                                          input_dim=args.hidden_dim_1,
                                          output_dim=args.hidden_dim_2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, args.hidden_dim_2], dtype=tf.float64) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=args.hidden_dim_2,
                                      act=tf.nn.tanh,
                                      logging=self.logging)(self.z)
