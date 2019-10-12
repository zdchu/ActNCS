from gcnetwork.gcmc.model import Model
# from crowdlayer.crowd_layer.crowd_layers import CrowdsClassification
import tensorflow as tf
from gcnetwork.gcmc.metrics import softmax_accuracy, expected_rmse, softmax_cross_entropy
from gcnetwork.gcmc.layers import *

from tensorflow.contrib.keras.api.keras.losses import categorical_crossentropy
from tf_crowdlayer import CrowdsClassification
import IPython

class SimpleGCNCrowd(Model):
    def __init__(self,  placeholders, input_dim, feat_hidden_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 num_side_features, self_connections=False, beta=0.5, conn_type='MW', **kwargs):
        super(SimpleGCNCrowd, self).__init__(**kwargs)
        self.inputs = (placeholders['u_features'], placeholders['v_features'])

        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']

        self.v_features_raw = placeholders['v_features_raw']

        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.responses = placeholders['responses']
        self.u_indices = placeholders['user_indices']
        self.labels = placeholders['labels']
        self.v_indices = placeholders['item_indices']
        self.class_values = placeholders['class_values']
        self.true_labels = placeholders['true_labels']
        self.beta = beta

        self.num_side_features = num_side_features
        self.feat_hidden_dim = feat_hidden_dim
        if num_side_features > 0:
            self.u_features_side = placeholders['u_features_side']
            self.v_features_side = placeholders['v_features_side']

        else:
            self.u_features_side = None
            self.v_features_side = None

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate
        self.conn_type = conn_type

        self.multiloss = 0
        self.gcnloss = 0

        self.build()
        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

    def _build(self):
        if self.accum == 'sum':
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                 output_dim=self.hidden[0],
                                                 support=self.support,
                                                 support_t=self.support_t,
                                                 num_support=self.num_support,
                                                 u_features_nonzero=self.u_features_nonzero,
                                                 v_features_nonzero=self.v_features_nonzero,
                                                 sparse_inputs=True,
                                                 act=tf.nn.relu,
                                                 bias=False,
                                                 dropout=self.dropout,
                                                 logging=self.logging,
                                                 share_user_item_weights=True,
                                                 self_connections=False))

        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging,
                                        share_user_item_weights=True))
        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        self.layers.append(Dense(input_dim=self.hidden[0],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=True))

        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           logging=self.logging,
                                           diagonal=False))

    def _single_loss(self):
        true_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.crowd_logits, labels=self.responses, dim=1)
        mask = tf.equal(self.responses[:, 0, :], -1)
        zeros = tf.zeros_like(true_loss_vec)
        loss = tf.where(mask, x=zeros, y=true_loss_vec)
        self.single_loss = tf.reduce_mean(loss)

    def _loss(self):
        true_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.compl_logits, labels=self.responses, dim=1)
        mask = tf.equal(self.responses[:, 0, :], -1)
        zeros = tf.zeros_like(true_loss_vec)
        fake_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.crowd_logits, labels=self.responses, dim=1)
        true_loss = tf.where(mask, x=zeros, y=true_loss_vec)
        fake_loss = tf.where(mask, x=zeros, y=fake_loss_vec)

        crowd_logits = tf.reshape(tf.transpose(self.crowd_logits, [0, 2, 1]), [-1, self.num_classes])
        rela_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=crowd_logits, labels=tf.reshape(self.compl_labels, [-1]))
        mask = tf.reshape(mask, [-1])
        zeros = tf.zeros_like(rela_loss_vec)
        rela_loss = tf.where(tf.logical_not(mask), x=zeros, y=rela_loss_vec)
        self.loss = tf.reduce_mean(true_loss) + tf.reduce_mean(fake_loss) \
                    + self.beta * tf.reduce_mean(rela_loss)

    def _alter_loss(self):
        true_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.compl_logits, labels=self.responses, dim=1)
        mask = tf.equal(self.responses[:, 0, :], -1)
        zeros = tf.zeros_like(true_loss_vec)
        true_loss = tf.where(mask, x=zeros, y=true_loss_vec)

        crowd_logits = tf.reshape(tf.transpose(self.crowd_logits, [0, 2, 1]), [-1, self.num_classes])
        rela_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=crowd_logits, labels=tf.reshape(self.compl_labels, [-1]))
        self.alter_loss = tf.reduce_mean(true_loss) + self.beta * tf.reduce_mean(rela_loss_vec)

    def _mse_loss(self):
        true_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.compl_logits, labels=self.responses, dim=1)
        mask = tf.equal(self.responses[:, 0, :], -1)
        zeros = tf.zeros_like(true_loss_vec)
        fake_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.crowd_logits, labels=self.responses, dim=1)
        true_loss = tf.where(mask, x=zeros, y=true_loss_vec)
        fake_loss = tf.where(mask, x=zeros, y=fake_loss_vec)

        crowd_logits = tf.reshape(tf.transpose(self.crowd_logits, [0, 2, 1]), [-1, self.num_classes])
        compl_logits = tf.reshape(tf.transpose(self.compl_logits, [0, 2, 1]), [-1, self.num_classes])
        mask = tf.equal(self.responses, -1)
        mask = tf.reshape(tf.transpose(mask, [0, 2, 1]), [-1, self.num_classes])
        # rela_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=crowd_logits, labels=tf.reshape(self.compl_labels, [-1]))
        zeros = tf.zeros_like(crowd_logits)
        crowd_loss = tf.where(tf.logical_not(mask), x=zeros, y=crowd_logits)
        compl_loss = tf.where(tf.logical_not(mask), x=zeros, y=compl_logits)
        self.mse_loss = tf.reduce_mean(true_loss) + tf.reduce_mean(fake_loss) \
                    + self.beta * tf.losses.mean_squared_error(crowd_loss, compl_loss)

    def _crowd_loss(self):
        crowd_logits = tf.reshape(tf.transpose(self.crowd_logits, [0, 2, 1]), [-1, self.num_classes])
        rela_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=crowd_logits, labels=tf.reshape(self.compl_labels, [-1]))
        # self.selected_mask = tf.reshape(self.selected_mask, [-1])
        # zeros = tf.zeros_like(rela_loss_vec)
        # rela_loss_vec = tf.where(self.selected_mask, x=zeros, y=rela_loss_vec)
        self.multiloss += tf.reduce_mean(rela_loss_vec)

    def _gcn_loss(self):
        self.gcnloss += softmax_cross_entropy(self.outputs, self.labels)

    def _normal_loss(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.basic_logits, labels=self.true_labels)
        self.normal_loss = loss

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('rmse_score', self.rmse)

    def basic_classifier(self, x):
        x = tf.keras.layers.Flatten(input_shape=x.shape[1:])(x)
        x = tf.keras.layers.Dense(1024, input_dim=x.shape, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, input_dim=x.shape, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, input_dim=x.shape, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.num_classes)(x)
        return x

    def build(self):
        with tf.variable_scope(self.name):
            self._build()
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1][1]
        self.outputs_all = self.activations[-1][0]

        self.compl_logits = tf.reshape(self.outputs_all, shape=(self.num_users, self.num_items, self.num_classes))
        self.compl_logits = tf.transpose(self.compl_logits, [1, 2, 0])

        prob = tf.nn.softmax(self.compl_logits, 1)
        rela_prob = prob[:, 0, :] - prob[:, 1, :]
        rela_prob = tf.abs(rela_prob)
        self.selected_mask = tf.less(rela_prob, 0)

        self.compl_labels = tf.argmax(self.compl_logits, 1)

        self.basic_logits = self.basic_classifier(self.v_features_raw)
        crowd_layer = CrowdsClassification(self.num_classes, self.num_users, conn_type=self.conn_type)
        self.crowd_logits = crowd_layer(self.basic_logits)

        self._crowd_loss()
        self._gcn_loss()
        self._alter_loss()
        self._mse_loss()
        self._single_loss()
        self._rmse()
        self._loss()
        self._normal_loss()



class GCNCrowd(Model):
    def __init__(self,  placeholders, input_dim, feat_hidden_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 num_side_features, self_connections=False, beta=0.5, conn_type='MW', **kwargs):
        super(GCNCrowd, self).__init__(**kwargs)
        self.inputs = (placeholders['u_features'], placeholders['v_features'])

        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']

        self.v_features_raw = placeholders['v_features_raw']

        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.responses = placeholders['responses']
        self.u_indices = placeholders['user_indices']
        self.labels = placeholders['labels']
        self.v_indices = placeholders['item_indices']
        self.class_values = placeholders['class_values']
        self.true_labels = placeholders['true_labels']
        self.beta = beta

        self.num_side_features = num_side_features
        self.feat_hidden_dim = feat_hidden_dim
        if num_side_features > 0:
            self.u_features_side = placeholders['u_features_side']
            self.v_features_side = placeholders['v_features_side']

        else:
            self.u_features_side = None
            self.v_features_side = None

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate
        self.conn_type = conn_type

        self.multiloss = 0
        self.gcnloss = 0

        self.build()

        # moving_average_decay = 0.995
        # self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        # self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

    def _single_loss(self):
        true_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.crowd_logits, labels=self.responses, dim=1)
        mask = tf.equal(self.responses[:, 0, :], -1)
        zeros = tf.zeros_like(true_loss_vec)
        loss = tf.where(mask, x=zeros, y=true_loss_vec)
        self.single_loss = tf.reduce_mean(loss)


    def _loss(self):
        true_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.compl_logits, labels=self.responses, dim=1)
        mask = tf.equal(self.responses[:, 0, :], -1)
        zeros = tf.zeros_like(true_loss_vec)
        fake_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.crowd_logits, labels=self.responses, dim=1)
        true_loss = tf.where(mask, x=zeros, y=true_loss_vec)
        fake_loss = tf.where(mask, x=zeros, y=fake_loss_vec)

        crowd_logits = tf.reshape(tf.transpose(self.crowd_logits, [0, 2, 1]), [-1, self.num_classes])
        rela_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=crowd_logits, labels=tf.reshape(self.compl_labels, [-1]))
        mask = tf.reshape(mask, [-1])
        zeros = tf.zeros_like(rela_loss_vec)
        rela_loss = tf.where(tf.logical_not(mask), x=zeros, y=rela_loss_vec)
        self.loss = tf.reduce_mean(true_loss) + tf.reduce_mean(fake_loss) \
                    + self.beta * tf.reduce_mean(rela_loss)

    def _masked_joint_loss(self):
        true_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.compl_logits, labels=self.responses, dim=1)
        mask = tf.equal(self.responses[:, 0, :], -1)
        zeros = tf.zeros_like(true_loss_vec)
        fake_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.crowd_logits, labels=self.responses, dim=1)
        true_loss = tf.where(mask, x=zeros, y=true_loss_vec)
        fake_loss = tf.where(mask, x=zeros, y=fake_loss_vec)

        crowd_logits = tf.reshape(tf.transpose(self.crowd_logits, [0, 2, 1]), [-1, self.num_classes])
        rela_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=crowd_logits, labels=tf.reshape(self.compl_labels, [-1]))
        mask = tf.reshape(mask, [-1])
        zeros = tf.zeros_like(rela_loss_vec)
        rela_loss = tf.where(tf.logical_not(mask), x=zeros, y=rela_loss_vec)

        self.selected_mask = tf.reshape(self.selected_mask, [-1])
        zeros = tf.zeros_like(rela_loss_vec)
        rela_loss = tf.where(self.selected_mask, x=zeros, y=rela_loss)
        self.masked_joint_loss = tf.reduce_mean(true_loss) + tf.reduce_mean(fake_loss) \
                    + self.beta * tf.reduce_mean(rela_loss)

    def _alter_loss(self):
        true_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.compl_logits, labels=self.responses, dim=1)
        mask = tf.equal(self.responses[:, 0, :], -1)
        zeros = tf.zeros_like(true_loss_vec)
        true_loss = tf.where(mask, x=zeros, y=true_loss_vec)

        crowd_logits = tf.reshape(tf.transpose(self.crowd_logits, [0, 2, 1]), [-1, self.num_classes])
        rela_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=crowd_logits, labels=tf.reshape(self.compl_labels, [-1]))
        self.alter_loss = tf.reduce_mean(true_loss) + self.beta * tf.reduce_mean(rela_loss_vec)

    def _mse_loss(self):
        true_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.compl_logits, labels=self.responses, dim=1)
        mask = tf.equal(self.responses[:, 0, :], -1)
        zeros = tf.zeros_like(true_loss_vec)
        fake_loss_vec = tf.nn.softmax_cross_entropy_with_logits(logits=self.crowd_logits, labels=self.responses, dim=1)
        true_loss = tf.where(mask, x=zeros, y=true_loss_vec)
        fake_loss = tf.where(mask, x=zeros, y=fake_loss_vec)

        crowd_logits = tf.reshape(tf.transpose(self.crowd_logits, [0, 2, 1]), [-1, self.num_classes])
        compl_logits = tf.reshape(tf.transpose(self.compl_logits, [0, 2, 1]), [-1, self.num_classes])
        mask = tf.equal(self.responses, -1)
        mask = tf.reshape(tf.transpose(mask, [0, 2, 1]), [-1, self.num_classes])
        # rela_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=crowd_logits, labels=tf.reshape(self.compl_labels, [-1]))
        zeros = tf.zeros_like(crowd_logits)
        crowd_loss = tf.where(tf.logical_not(mask), x=zeros, y=crowd_logits)
        compl_loss = tf.where(tf.logical_not(mask), x=zeros, y=compl_logits)
        self.mse_loss = tf.reduce_mean(true_loss) + tf.reduce_mean(fake_loss) \
                    + self.beta * tf.losses.mean_squared_error(crowd_loss, compl_loss)

    def _crowd_loss(self):
        crowd_logits = tf.reshape(tf.transpose(self.crowd_logits, [0, 2, 1]), [-1, self.num_classes])
        rela_loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=crowd_logits, labels=tf.reshape(self.compl_labels, [-1]))
        self.selected_mask = tf.reshape(self.selected_mask, [-1])
        zeros = tf.zeros_like(rela_loss_vec)
        rela_loss_vec = tf.where(self.selected_mask, x=zeros, y=rela_loss_vec)
        self.multiloss += tf.reduce_mean(rela_loss_vec)

    def _gcn_loss(self):
        self.gcnloss += softmax_cross_entropy(self.outputs, self.labels)

    def _normal_loss(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.basic_logits, labels=self.true_labels)
        self.normal_loss = loss

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('rmse_score', self.rmse)

    def _build(self):
        if self.accum == 'sum':
            self.layers.append(
                OrdinalMixtureGCN(input_dim=self.input_dim,
                                  output_dim=self.hidden[0],
                                  support=self.support,
                                  support_t=self.support_t,
                                  num_support=self.num_support,
                                  u_features_nonzero=self.u_features_nonzero,
                                  v_features_nonzero=self.v_features_nonzero,
                                  sparse_inputs=True,
                                  act=tf.nn.relu,
                                  bias=False,
                                  dropout=self.dropout,
                                  logging=self.logging,
                                  share_user_item_weights=True,
                                  self_connections=self.self_connections))
        elif self.accum == 'stack':
            self.layers.append(StackGCN(
                input_dim=self.input_dim,
                output_dim=self.hidden[0],
                support=self.support,
                support_t=self.support_t,
                num_support=self.num_support,
                u_features_nonzero=self.u_features_nonzero,
                v_features_nonzero=self.v_features_nonzero,
                sparse_inputs=True,
                act=tf.nn.relu,
                dropout=self.dropout,
                logging=self.logging,
                share_user_item_weights=True)
            )
        else:
            raise ValueError('accumulation function option invalid')

        self.layers.append(Dense(input_dim=self.num_side_features,
                                 output_dim=self.feat_hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=0.,
                                 logging=self.logging,
                                 bias=True,
                                 share_user_item_weights=False))

        self.layers.append(Dense(
            input_dim=self.hidden[0] + self.feat_hidden_dim,
            output_dim=self.hidden[1],
            act=lambda x: x,
            dropout=self.dropout,
            logging=self.logging,
            share_user_item_weights=False
        ))

        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           logging=self.logging,
                                           diagonal=False)
                           )

    def basic_classifier(self, x):
        x = tf.keras.layers.Flatten(input_shape=x.shape[1:])(x)
        x = tf.keras.layers.Dense(1024, input_dim=x.shape, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, input_dim=x.shape, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, input_dim=x.shape, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.num_classes)(x)
        return x

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        layer = self.layers[0]
        gcn_hidden = layer(self.inputs)

        layer = self.layers[1]
        feat_hidden = layer([self.u_features_side, self.v_features_side])

        layer = self.layers[2]

        gcn_u = gcn_hidden[0]
        gcn_v = gcn_hidden[1]
        feat_u = feat_hidden[0]
        feat_v = feat_hidden[1]

        input_u = tf.concat(values=[gcn_u, feat_u], axis=1)
        input_v = tf.concat(values=[gcn_v, feat_v], axis=1)

        concat_hidden = layer([input_u, input_v])

        self.activations.append(concat_hidden)

        for layer in self.layers[3::]:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        self.outputs = self.activations[-1][1]
        self.outputs_all = self.activations[-1][0] # completed matrix
        self.compl_logits = tf.reshape(self.outputs_all, shape=(self.num_users, self.num_items, self.num_classes))

        self.compl_logits = tf.transpose(self.compl_logits, [1, 2, 0])

        prob = tf.nn.softmax(self.compl_logits, 1)
        pred_probs = tf.reduce_max(prob, axis=1)
        self.selected_mask = tf.less(pred_probs, 0.8)

        self.compl_labels = tf.argmax(self.compl_logits, 1)

        self.basic_logits = self.basic_classifier(self.v_features_raw)
        crowd_layer = CrowdsClassification(self.num_classes, self.num_users, conn_type=self.conn_type)
        self.crowd_logits = crowd_layer(self.basic_logits)

        self._crowd_loss()
        self._gcn_loss()
        self._alter_loss()
        self._mse_loss()
        self._single_loss()
        self._rmse()
        self._loss()
        self._masked_joint_loss()
        self._normal_loss()
