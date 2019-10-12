from keras.layers import *
import tensorflow as tf
from gcnetwork.gcmc.initializations import *

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)

    return pre_out * tf.div(1., keep_prob)

class OrdinalMixtureGCN(Layer):
    """Graph convolution layer for bipartite graphs and sparse inputs."""
    def __init__(self, input_dim, output_dim, support, support_t, num_support, u_features_nonzero=None,
                 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
                 act=tf.nn.relu, bias=False, share_user_item_weights=False, self_connections=False, **kwargs):
        self.vars = {}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_support = num_support
        self.bias = bias
        self.share_user_item_weights = share_user_item_weights
        self.dropout = dropout
        self.u_features_nonzero = u_features_nonzero
        self.v_features_nonzero = v_features_nonzero
        self.sparse_inputs = sparse_inputs
        self.act = act
        self.self_connections = self_connections
        self.support = tf.sparse_split(axis=1, num_split=self.num_support, sp_input=support)
        self.support_t = tf.sparse_split(axis=1, num_split=self.num_support, sp_input=support_t)
        super(OrdinalMixtureGCN, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights_u'] = tf.stack([weight_variable_random_uniform(self.input_dim,
                                                                              self.output_dim,
                                                                             name='weights_u_%d' % i)
                                              for i in range(self.num_support)], axis=0)

            if self.bias:
                self.vars['bias_u'] = bias_variable_const([self.output_dim], 0.01, name="bias_u")

            if not self.share_user_item_weights:
                self.vars['weights_v'] = tf.stack([weight_variable_random_uniform(self.input_dim, self.output_dim,
                                                                                 name='weights_v_%d' % i)
                                                  for i in range(self.num_support)], axis=0)

                if self.bias:
                    self.vars['bias_v'] = bias_variable_const([self.output_dim], 0.01, name="bias_v")

            else:
                self.vars['weights_v'] = self.vars['weights_u']
                if self.bias:
                    self.vars['bias_v'] = self.vars['bias_u']

        self.weights_u = self.vars['weights_u']
        self.weights_v = self.vars['weights_v']
        if self.sparse_inputs:
            assert self.u_features_nonzero is not None and self.v_features_nonzero is not None, \
                'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'

        if self.self_connections:
            self.support = self.support[:-1]
            self.support_transpose = self.support_t[:-1]
            self.u_self_connections = self.support[-1]
            self.v_self_connections = self.support_t[-1]
            self.weights_u = self.weights_u[:-1]
            self.weights_v = self.weights_v[:-1]
            self.weights_u_self_conn = self.weights_u[-1]
            self.weights_v_self_conn = self.weights_v[-1]

        else:
            self.support = self.support
            self.support_transpose = self.support_t
            self.u_self_connections = None
            self.v_self_connections = None
            self.weights_u_self_conn = None
            self.weights_v_self_conn = None

        self.support_nnz = []
        self.support_transpose_nnz = []
        for i in range(len(self.support)):
            nnz = tf.reduce_sum(tf.shape(self.support[i].values))
            self.support_nnz.append(nnz)
            self.support_transpose_nnz.append(nnz)
        super(OrdinalMixtureGCN, self).build(input_shape)


    def _call(self, inputs):
        if self.sparse_inputs:
            x_u = dropout_sparse(inputs[0], 1 - self.dropout, self.u_features_nonzero)
            x_v = dropout_sparse(inputs[1], 1 - self.dropout, self.v_features_nonzero)
        else:
            x_u = tf.nn.dropout(inputs[0], 1 - self.dropout)
            x_v = tf.nn.dropout(inputs[1], 1 - self.dropout)
        supports_u = []
        supports_v = []
        # self-connections with identity matrix as support
        if self.self_connections:
            uw = dot(x_u, self.weights_u_self_conn, sparse=self.sparse_inputs)
            supports_u.append(tf.sparse_tensor_dense_matmul(self.u_self_connections, uw))

            vw = dot(x_v, self.weights_v_self_conn, sparse=self.sparse_inputs)
            supports_v.append(tf.sparse_tensor_dense_matmul(self.v_self_connections, vw))

        wu = 0.
        wv = 0.
        for i in range(len(self.support)):
            wu += self.weights_u[i]
            wv += self.weights_v[i]

            # multiply feature matrices with weights
            tmp_u = dot(x_u, wu, sparse=self.sparse_inputs)

            tmp_v = dot(x_v, wv, sparse=self.sparse_inputs)

            support = self.support[i]
            support_transpose = self.support_transpose[i]

            # then multiply with rating matrices
            supports_u.append(tf.sparse_tensor_dense_matmul(support, tmp_v))
            supports_v.append(tf.sparse_tensor_dense_matmul(support_transpose, tmp_u))

        z_u = tf.add_n(supports_u)
        z_v = tf.add_n(supports_v)

        if self.bias:
            z_u = tf.nn.bias_add(z_u, self.vars['bias_u'])
            z_v = tf.nn.bias_add(z_v, self.vars['bias_v'])

        u_outputs = ReLU()(z_u)
        v_outputs = ReLU()(z_v)

        return u_outputs, v_outputs

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs[0].shape)
            self.built = True
        return self.call(inputs)

    def call(self, inputs):
        with tf.name_scope(self.name):
            outputs_u, outputs_v = self._call(inputs)
            return outputs_u, outputs_v
