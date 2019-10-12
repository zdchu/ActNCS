import numpy as np
import tensorflow as tf

def init_identities(shape, dtype=None):
	out = np.zeros(shape)
	for r in range(shape[2]):
		for i in range(shape[0]):
			out[i,i,r] = 2.0
	return out

class CrowdsClassification(object):
    def __init__(self, output_dim, num_annotators, var_scope=None, conn_type="MW", **kwargs):
        self.output_dim = output_dim
        self.num_annotators = num_annotators
        self.conn_type = conn_type

        super(CrowdsClassification, self).__init__(**kwargs)
        with tf.variable_scope(var_scope or 'crowd_layer', reuse=tf.AUTO_REUSE):
            if self.conn_type == 'MW':
                init = tf.constant(init_identities([self.output_dim, self.output_dim, self.num_annotators]), dtype=tf.float32)
                self.kernel = tf.get_variable('weight_matrix', initializer=init, dtype=tf.float32)
            elif self.conn_type == 'VW':
                self.kernel = tf.get_variable('weight_vector', shape=[self.output_dim, self.num_annotators],
                                              initializer=tf.initializers.ones())
            elif self.conn_type == 'VB':
                self.kernel = tf.get_variable('weight', shape=(self.output_dim, self.num_annotators),
                                                   initializer=tf.initializers.ones())
            elif self.conn_type == 'VW+B':
                self.kernel_w = tf.get_variable('weight_vector', shape=(self.output_dim, self.num_annotators),
                                              initializer=tf.initializers.ones())
                self.kernel_b = tf.get_variable('bias_vector', shape=(self.output_dim, self.num_annotators),
                                                initializer=tf.initializers.zeros())
            elif self.conn_type == 'SW':
                self.kernel = tf.get_variable('weight', shape=(self.num_annotators, 1),
                                              initializer=tf.initializers.ones())

    def __call__(self, x):
        if self.conn_type == 'MW':
            res = tf.keras.backend.dot(x, self.kernel)
        elif self.conn_type == "VW" or self.conn_type == "VB" or self.conn_type == "VW+B" or self.conn_type == "SW":
            out = []
            for r in range(self.num_annotators):
                if self.conn_type == "VW":
                    out.append(x * self.kernel[:,r])
                elif self.conn_type == "VB":
                    out.append(x + self.kernel[:,r])
                elif self.conn_type == "VW+B":
                    out.append(x * self.kernel_w[:,r] + self.kernel_b[:,r])
                elif self.conn_type == "SW":
                    out.append(x * self.kernel[r, 0])
            res = tf.stack(out)
            if len(res.shape) == 3:
                res = tf.transpose(res, [1, 2, 0])
            elif len(res.shape) == 4:
                res = tf.transpose(res, [1, 2, 3, 0])
        return res
