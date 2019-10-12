from __future__ import division
from __future__ import print_function
import argparse
from dataUtil import feed_dict_4_GCN
import opts

import IPython
import tensorflow as tf
from collections import Counter
import sys
from copy import deepcopy

from new_dataUtil import *
from evalUtil import *

from joint_model import GCNCrowd, SimpleGCNCrowd
from sklearn.utils.extmath import softmax
from sklearn.metrics import accuracy_score

from gcnetwork.gcmc.preprocessing import *

# packages for learning from crowds
from crowdlayer.crowd_layer.crowd_layers import CrowdsClassification, MaskedMultiCrossEntropy

from gcnetwork.gcmc.preprocessing import create_trainvaltest_split, \
    sparse_to_tuple, preprocess_user_item_features, globally_normalize_bipartite_adjacency, \
    load_data_monti, load_official_trainvaltest_split, normalize_features, load_data_sensor, load_data_crowd
from gcnetwork.gcmc.model import RecommenderGAE, RecommenderSideInfoGAE
from gcnetwork.gcmc.utils import construct_feed_dict

from matrix_completion import matrix_completion_NMF
from labelAggregation import IWMV
'''
GCN-based matrix completion
'''

np.random.seed(1234)
tf.set_random_seed(1234)


def GCN_with_test(u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, class_values,
                  test_u_indices, test_v_indices, test_labels, val_u_indices, val_v_indices, val_labels
                  ):
    num_users, num_items = adj_train.shape
    num_side_features = 0
    if not opt.features:
        u_features = sp.identity(num_users, format='csr')
        v_features = sp.identity(num_items, format='csr')

        u_features, v_features = preprocess_user_item_features(u_features, v_features)
    elif opt.features and u_features is not None and v_features is not None:
        print('Normalizing feature vectors...')
        u_features_side = normalize_features(u_features)
        v_features_side = normalize_features(v_features)

        u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

        u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
        v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

        num_side_features = u_features_side.shape[1]
        id_csr_v = sp.identity(num_items, format='csr')
        id_csr_u = sp.identity(num_users, format='csr')

        u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)
    else:
        raise ValueError('Features error')

    support = []
    support_t = []
    adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)
    for i in range(len(class_values)):
        support_unnormalized = sp.csr_matrix(adj_train_int==i + 1, dtype=np.float32)
        if support_unnormalized.nnz == 0:
            sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

        support_unnormalized_transpose = support_unnormalized.T
        support.append(support_unnormalized)
        support_t.append(support_unnormalized_transpose)
    support = globally_normalize_bipartite_adjacency(support, symmetric=opt.norm_symmetric)
    support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=opt.norm_symmetric)

    if opt.self_connections:
        support.append(sp.identity(u_features.shape[0], format='csr'))
        support_t.append(sp.identity(v_features.shape[0], format='csr'))

    num_support = len(support)
    support = sp.hstack(support, format='csr')
    support_t = sp.hstack(support_t, format='csr')

    if opt.accumulation == 'stack':
        div = opt.hidden[0] // num_support
        if opt.hidden[0] % num_support != 0:
            print("""\nWARNING: opt.hidden[0] (=%d) of stack layer is adjusted to %d such that
                  it can be evenly split in %d splits.\n""" % (opt.hidden[0], num_support * div, num_support))
        opt.hidden[0] = num_support * div
    test_u = list(set(test_u_indices))
    test_v = list(set(test_v_indices))
    test_u_dict = {n:i for i, n in enumerate(test_u)}
    test_v_dict = {n:i for i, n in enumerate(test_v)}

    test_u_indices = np.array([test_u_dict[o] for o in test_u_indices])
    test_v_indices = np.array([test_v_dict[o] for o in test_v_indices])

    test_support = support[np.array(test_u)]
    test_support_t = support_t[np.array(test_v)]

    val_u = list(set(val_u_indices))
    val_v = list(set(val_v_indices))
    val_u_dict = {n: i for i, n in enumerate(val_u)}
    val_v_dict = {n: i for i, n in enumerate(val_v)}

    val_u_indices = np.array([val_u_dict[o] for o in val_u_indices])
    val_v_indices = np.array([val_v_dict[o] for o in val_v_indices])

    val_support = support[np.array(val_u)]
    val_support_t = support_t[np.array(val_v)]

    train_u = list(set(train_u_indices))
    train_v = list(set(train_v_indices))
    train_u_dict = {n: i for i, n in enumerate(train_u)}
    train_v_dict = {n: i for i, n in enumerate(train_v)}

    train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
    train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])

    train_support = support[np.array(train_u)]
    train_support_t = support_t[np.array(train_v)]

    if opt.features:
        test_u_features_side = u_features_side[np.array(test_u)]
        test_v_features_side = v_features_side[np.array(test_v)]

        val_u_features_side = u_features_side[np.array(val_u)]
        val_v_features_side = v_features_side[np.array(val_v)]

        train_u_features_side = u_features_side[np.array(train_u)]
        train_v_features_side = v_features_side[np.array(train_v)]
    else:
        test_u_features_side = None
        test_v_features_side = None

        val_u_features_side = None
        val_v_features_side = None

        train_u_features_side = None
        train_v_features_side = None

    placeholders = {
        'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
        'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
        'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
        'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
        'labels': tf.placeholder(tf.int32, shape=(None,)),

        'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
        'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

        'user_indices': tf.placeholder(tf.int32, shape=(None,)),
        'item_indices': tf.placeholder(tf.int32, shape=(None,)),

        'class_values': tf.placeholder(tf.float32, shape=class_values.shape),

        'dropout': tf.placeholder_with_default(0., shape=()),
        'weight_decay': tf.placeholder_with_default(0., shape=()),

        'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
        'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    }

    if opt.features:
        model = RecommenderSideInfoGAE(placeholders,
                                       input_dim=u_features.shape[1],
                                       feat_hidden_dim=opt.feat_hidden,
                                       num_classes=len(class_values),
                                       num_support=num_support,
                                       self_connections=opt.self_connections,
                                       num_basis_functions=opt.num_basis_functions,
                                       hidden=opt.hidden,
                                       num_users=num_users,
                                       num_items=num_items,
                                       accum=opt.accumulation,
                                       learning_rate=opt.learning_rate,
                                       num_side_features=num_side_features,
                                       logging=True)
    else:
        model = RecommenderGAE(placeholders,
                               input_dim=u_features.shape[1],
                               num_classes=len(class_values),
                               num_support=num_support,
                               self_connections=opt.self_connections,
                               num_basis_functions=opt.num_basis_functions,
                               hidden=opt.hidden,
                               num_users=num_users,
                               num_items=num_items,
                               accum=opt.accumulation,
                               learning_rate=opt.learning_rate,
                               logging=True)
    test_support = sparse_to_tuple(test_support)
    test_support_t = sparse_to_tuple(test_support_t)

    val_support = sparse_to_tuple(val_support)
    val_support_t = sparse_to_tuple(val_support_t)

    train_support = sparse_to_tuple(train_support)
    train_support_t = sparse_to_tuple(train_support_t)

    u_features = sparse_to_tuple(u_features)
    v_features = sparse_to_tuple(v_features)
    assert u_features[2][1] == v_features[2][1]

    num_features = u_features[2][1]
    u_features_nonzero = u_features[1].shape[0]
    v_features_nonzero = v_features[1].shape[0]

    train_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                      v_features_nonzero, train_support, train_support_t,
                                      train_labels, train_u_indices, train_v_indices, class_values, opt.dropout,
                                      train_u_features_side, train_v_features_side)

    val_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                    v_features_nonzero, val_support, val_support_t,
                                    val_labels, val_u_indices, val_v_indices, class_values, 0.,
                                    val_u_features_side, val_v_features_side)

    test_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                     v_features_nonzero, test_support, test_support_t,
                                     test_labels, test_u_indices, test_v_indices, class_values, 0.,
                                     test_u_features_side, test_v_features_side)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    best_val_score = np.inf

    print('Training...')
    for epoch in range(opt.epochs):
        outs = sess.run([model.training_op, model.loss, model.rmse], feed_dict=train_feed_dict)

        train_avg_loss = outs[1]
        train_rmse = outs[2]

        val_avg_loss, val_rmse, val_acc = sess.run([model.loss, model.rmse, model.accuracy], feed_dict=val_feed_dict)

        if val_rmse < best_val_score:
            best_val_score = val_rmse
            best_epoch = epoch

        if opt.verbose:
            print("[*] Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_loss),
              "train_rmse=", "{:.5f}".format(train_rmse),
              "val_loss=", "{:.5f}".format(val_avg_loss),
              "val_rmse=", "{:.5f}".format(val_rmse),
            'val_acc=', '{:.5f}'.format(val_acc))

    if opt.testing:
        test_avg_loss, test_rmse, test_acc, response_test = sess.run([model.loss, model.rmse, model.accuracy, model.outputs_all], feed_dict=test_feed_dict)
        print('test loss = ', test_avg_loss)
        print('test rmse = ', test_rmse)
        print('test acc = ', test_acc)

    response_train = sess.run(model.outputs_all, feed_dict=train_feed_dict)
    probs = softmax(response_train)
    pred_labels = np.argmax(probs, axis=1)
    pred_labels = pred_labels.reshape((num_users, num_items))
    return transform_onehot(pred_labels.T, num_users, len(class_values))


def matrix_completion_GCN(u_features, v_features, adj_train, train_labels, train_u_indices,
                          train_v_indices, class_values, threshold):
    '''
    :param u_features:
    :param v_features:
    :param adj_train: csr_matrix, sparse rating_matrix_train, size = [M, N], elem = response inner idx + 1: 1, 2, 3,...
    :param train_labels: |observed in train|, elem = 0,1,2,...
    :param train_u_indices: size = |observed response in train|
    :param train_v_indices: size = |observed response in train|
    :param class_values:
    :param trn_instance_idx: size = |trn instance|
    :return:
    '''
    '''
    filter observed
    '''
    # oracles
    num_users, num_items = adj_train.shape
    num_side_features = 0
    if not opt.features:
        u_features = sp.identity(num_users, format='csr')
        v_features = sp.identity(num_items, format='csr')

        u_features, v_features = preprocess_user_item_features(u_features, v_features)
    elif opt.features and u_features is not None and v_features is not None:
        print('Normalizing feature vectors...')
        u_features_side = normalize_features(u_features)
        v_features_side = normalize_features(v_features)

        u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

        u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
        v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

        num_side_features = u_features_side.shape[1]
        id_csr_v = sp.identity(num_items, format='csr')
        id_csr_u = sp.identity(num_users, format='csr')

        u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)
    else:
        raise ValueError('Features error')

    support = []
    support_t = []
    adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)
    for i in range(len(class_values)):
        support_unnormalized = sp.csr_matrix(adj_train_int==i + 1, dtype=np.float32)
        if support_unnormalized.nnz == 0:
            sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

        support_unnormalized_transpose = support_unnormalized.T
        support.append(support_unnormalized)
        support_t.append(support_unnormalized_transpose)
    support = globally_normalize_bipartite_adjacency(support, symmetric=opt.norm_symmetric)
    support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=opt.norm_symmetric)

    if opt.self_connections:
        support.append(sp.identity(u_features.shape[0], format='csr'))
        support_t.append(sp.identity(v_features.shape[0], format='csr'))

    num_support = len(support)
    support = sp.hstack(support, format='csr')
    support_t = sp.hstack(support_t, format='csr')

    if opt.accumulation == 'stack':
        div = opt.hidden[0] // num_support
        if opt.hidden[0] % num_support != 0:
            print("""\nWARNING: opt.hidden[0] (=%d) of stack layer is adjusted to %d such that
                  it can be evenly split in %d splits.\n""" % (opt.hidden[0], num_support * div, num_support))
        opt.hidden[0] = num_support * div
    train_u = list(set(train_u_indices))
    train_v = list(set(train_v_indices))
    train_u_dict = {n: i for i, n in enumerate(train_u)}
    train_v_dict = {n: i for i, n in enumerate(train_v)}

    train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
    train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])

    train_support = support[np.array(train_u)]
    train_support_t = support_t[np.array(train_v)]

    if opt.features:
        train_u_features_side = u_features_side[np.array(train_u)]
        train_v_features_side = v_features_side[np.array(train_v)]
    else:
        train_u_features_side = None
        train_v_features_side = None

    placeholders = {
        'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
        'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
        'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
        'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
        'labels': tf.placeholder(tf.int32, shape=(None,)),

        'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
        'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

        'user_indices': tf.placeholder(tf.int32, shape=(None,)),
        'item_indices': tf.placeholder(tf.int32, shape=(None,)),

        'class_values': tf.placeholder(tf.float32, shape=class_values.shape),

        'dropout': tf.placeholder_with_default(0., shape=()),
        'weight_decay': tf.placeholder_with_default(0., shape=()),

        'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
        'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    }

    if opt.features:
        model = RecommenderSideInfoGAE(placeholders,
                                       input_dim=u_features.shape[1],
                                       feat_hidden_dim=opt.feat_hidden,
                                       num_classes=len(class_values),
                                       num_support=num_support,
                                       self_connections=opt.self_connections,
                                       num_basis_functions=opt.num_basis_functions,
                                       hidden=opt.hidden,
                                       num_users=num_users,
                                       num_items=num_items,
                                       accum=opt.accumulation,
                                       learning_rate=opt.learning_rate,
                                       num_side_features=num_side_features,
                                       logging=True)
    else:
        model = RecommenderGAE(placeholders,
                               input_dim=u_features.shape[1],
                               num_classes=len(class_values),
                               num_support=num_support,
                               self_connections=opt.self_connections,
                               num_basis_functions=opt.num_basis_functions,
                               hidden=opt.hidden,
                               num_users=num_users,
                               num_items=num_items,
                               accum=opt.accumulation,
                               learning_rate=opt.learning_rate,
                               logging=True)

    train_support = sparse_to_tuple(train_support)
    train_support_t = sparse_to_tuple(train_support_t)

    u_features = sparse_to_tuple(u_features)
    v_features = sparse_to_tuple(v_features)
    assert u_features[2][1] == v_features[2][1]

    num_features = u_features[2][1]
    u_features_nonzero = u_features[1].shape[0]
    v_features_nonzero = v_features[1].shape[0]

    train_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                      v_features_nonzero, train_support, train_support_t,
                                      train_labels, train_u_indices, train_v_indices, class_values, opt.dropout,
                                      train_u_features_side, train_v_features_side)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Training...')
    for epoch in range(opt.epochs):
        outs = sess.run([model.training_op, model.loss, model.rmse], feed_dict=train_feed_dict)

        train_avg_loss = outs[1]
        train_rmse = outs[2]

        # val_avg_loss, val_rmse, val_acc = sess.run([model.loss, model.rmse, model.accuracy], feed_dict=val_feed_dict)

        if opt.verbose:
            print("[*] Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_loss),
              "train_rmse=", "{:.5f}".format(train_rmse))

    if opt.testing:
        test_avg_loss, test_rmse, test_acc, response_test = sess.run([model.loss, model.rmse,
                                                                      model.accuracy, model.outputs_all], feed_dict=test_feed_dict)
        print('test loss = ', test_avg_loss)
        print('test rmse = ', test_rmse)
        print('test acc = ', test_acc)
    response_train = sess.run(model.outputs_all, feed_dict=train_feed_dict)
    probs = softmax(response_train)
    # diff = abs(probs[:, 0] - probs[:, 1])
    pred_labels = np.argmax(probs, axis=1)
    pred_probs = np.max(probs, axis=1)
    # IPython.embed()
    pred_labels[pred_probs < threshold] = -1
    pred_labels = pred_labels.reshape((num_users, num_items))

    # IPython.embed()
    pred_labels = pred_labels.T

    adj_train = adj_train - 1
    adj_train = adj_train.T
    '''
    for i, res in enumerate(pred_labels):
        exsit_labels = set(adj_train[i])
        zeros = np.zeros_like(res)
        for j in exsit_labels:
            zeros += res == j
        pred_labels[i][np.logical_not(zeros)] = -1
    '''
    return pred_labels, transform_onehot(pred_labels, num_users, len(class_values))


def train(opt):
    # datasplit_path = "./data/sensor/complete/"
    datasplit_path = './data/labelme'
    fold = 1
    N_EPOCHS = 300
    BATCH_SIZE = 100

    u_features, v_features, responses, class_values, \
    true_class, rating_mx_train, train_labels, train_u_indices, train_v_indices, \
    v_features_tst, true_class_tst = read_npy(datasplit_path, test_rate=opt.test_rate)

    # transform
    num_classes = len(class_values)
    num_oracles = u_features.shape[0]
    dim_instance = v_features.shape[1]

    IPython.embed()
    if opt.train_method == 'joint':
        placeholders, train_feed_dict, input_dim, num_users, num_items, num_support,\
            num_side_features, hidden_dim, responses_train = feed_dict_4_GCN(u_features, v_features, rating_mx_train, class_values,
                train_u_indices, train_v_indices, train_labels, self_connections=opt.self_connections, hidden_dim=opt.hidden[0], with_features=opt.features)
        opt.hidden[0] = hidden_dim

        true_labels = one_hot(true_class, num_classes)
        placeholders.update({
            'v_features_raw': tf.placeholder(tf.float32, shape=v_features.shape),
            'responses': tf.placeholder(tf.float32, shape=responses.shape),
            'true_labels': tf.placeholder(tf.float32, shape=true_labels.shape)
        })

        train_feed_dict.update(
            {placeholders['v_features_raw']: v_features.toarray(),
             placeholders['responses']: responses,
             placeholders['true_labels']: true_labels}
        )

        # IPython.embed()
        if opt.features:
            model = GCNCrowd(placeholders, input_dim=input_dim,
                         feat_hidden_dim=opt.feat_hidden,
                         num_classes=num_classes,
                         self_connections=opt.self_connections,
                         num_support=num_support,
                         num_basis_functions=opt.num_basis_functions,
                         hidden=opt.hidden,
                         num_users=num_users,
                         num_items=num_items,
                         accum=opt.accumulation,
                         num_side_features=num_side_features,
                         learning_rate=opt.learning_rate,
                         logging=True,
                         beta=opt.beta,
                         conn_type=opt.conn_type
                         )
        else:
            model = SimpleGCNCrowd(placeholders, input_dim=input_dim,
                             feat_hidden_dim=opt.feat_hidden,
                             num_classes=num_classes,
                             self_connections=opt.self_connections,
                             num_support=num_support,
                             num_basis_functions=opt.num_basis_functions,
                             hidden=opt.hidden,
                             num_users=num_users,
                             num_items=num_items,
                             accum=opt.accumulation,
                             num_side_features=num_side_features,
                             learning_rate=opt.learning_rate,
                             logging=True,
                             beta=opt.beta
                             )

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(opt.learning_rate, global_step, 200, 0.95, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=opt.learning_rate, beta1=0.9, beta2=0.999)
        print('Joint training...')

        moving_average_decay = 0.95
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        sess = tf.Session()
        normal_op = optimizer.minimize(model.masked_joint_loss, global_step=global_step)

        with tf.control_dependencies([normal_op]):
            opt_op = tf.group(variables_averages_op)

        sess.run(tf.global_variables_initializer())
        best_acc = 0
        best_f1 = 0
        best_epoch = 0

        val_feed_dict = train_feed_dict.copy()
        val_feed_dict[placeholders['dropout']] = 0
        for epoch in range(opt.epochs):
            outputs = sess.run([model.masked_joint_loss, model.basic_logits, opt_op, model.rmse], feed_dict=train_feed_dict)
            val_loss, val_pred, val_rmse = sess.run([model.masked_joint_loss, model.basic_logits, model.rmse], feed_dict=val_feed_dict)
            pred_labels = np.argmax(val_pred, axis=1)
            acc = accuracy_score(true_class, pred_labels)
            f1 = f1_score(true_class, pred_labels, average='macro')
            acc_per_class = accuracy_per_class(y_pred=pred_labels, y_true=true_class)
            print('[*] Epoch:', '%d' % (epoch + 1),
                  'train_loss=', '{:.5f}'.format(outputs[0]),
                  'val_loss=', '{:.5f}'.format(val_loss),
                  'val_acc=', '{:.5f}'.format(acc),
                  'val_f1=', '{:.5f}'.format(f1))
            if f1 > best_f1:
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch
                best_acc_per_class = acc_per_class
        print('[Best] Epoch:', '%d' % best_epoch,
                  'val_acc=', '{:.5f}'.format(best_acc),
                  'val_f1=', '{:.5f}'.format(best_f1))
        print('acc_per_class= ', best_acc_per_class)

        val_loss, val_pred, val_res = sess.run([model.loss, model.basic_logits],
                                                feed_dict=val_feed_dict)
        pred_labels = np.argmax(val_pred, axis=1)
        acc = accuracy_score(true_class, pred_labels)
        f1 = f1_score(true_class, pred_labels, average='macro')
        acc_per_class = accuracy_per_class(y_pred=pred_labels, y_true=true_class)
        print('test_acc=', '{:.5f}'.format(acc),
              'test_f1=', '{:.5f}'.format(f1))
        print('acc_per_class= ', acc_per_class)

        probs = softmax(val_res)
        pred_labels = np.argmax(probs, axis=1)
        pred_labels = pred_labels.reshape((num_users, num_items))
        responses_gcn = transform_onehot(pred_labels.T, num_users, len(class_values))
        predictY, weight, id_conf = IWMV(responses_gcn, class_values, dict())
        acc_iwmv_trn = accuracy_score(predictY, true_class)
        acc_iwmv_per_cls = accuracy_per_class(true_class, predictY)
        f1_iwmv = f1_score(true_class, predictY, average='macro')
        accuracy_trn = (acc_iwmv_trn, f1_iwmv, acc_iwmv_per_cls)
        print(accuracy_trn)

    elif opt.train_method == 'alternate':
        placeholders, train_feed_dict, input_dim, num_users, num_items, num_support, \
        num_side_features, hidden_dim, responses_train = feed_dict_4_GCN(u_features, v_features, rating_mx_train,
                                                                         class_values,
                                                                         train_u_indices, train_v_indices, train_labels,
                                                                         self_connections=opt.self_connections,
                                                                         hidden_dim=opt.hidden[0],
                                                                         with_features=opt.features)
        opt.hidden[0] = hidden_dim

        true_labels = one_hot(true_class, num_classes)
        placeholders.update({
            'v_features_raw': tf.placeholder(tf.float32, shape=v_features.shape),
            'responses': tf.placeholder(tf.float32, shape=responses.shape),
            'true_labels': tf.placeholder(tf.float32, shape=true_labels.shape)
        })

        train_feed_dict.update(
            {placeholders['v_features_raw']: v_features.toarray(),
             placeholders['responses']: responses,
             placeholders['true_labels']: true_labels}
        )

        if opt.features:
            model = GCNCrowd(placeholders, input_dim=input_dim,
                         feat_hidden_dim=opt.feat_hidden,
                         num_classes=num_classes,
                         self_connections=opt.self_connections,
                         num_support=num_support,
                         num_basis_functions=opt.num_basis_functions,
                         hidden=opt.hidden,
                         num_users=num_users,
                         num_items=num_items,
                         accum=opt.accumulation,
                         num_side_features=num_side_features,
                         learning_rate=opt.learning_rate,
                         logging=True,
                         beta=opt.beta
                         )
        else:
            model = SimpleGCNCrowd(placeholders, input_dim=input_dim,
                             feat_hidden_dim=opt.feat_hidden,
                             num_classes=num_classes,
                             self_connections=opt.self_connections,
                             num_support=num_support,
                             num_basis_functions=opt.num_basis_functions,
                             hidden=opt.hidden,
                             num_users=num_users,
                             num_items=num_items,
                             accum=opt.accumulation,
                             num_side_features=num_side_features,
                             learning_rate=opt.learning_rate,
                             logging=True,
                             beta=opt.beta
                             )
        sess = tf.Session()

        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer_1 = tf.train.AdamOptimizer(learning_rate=opt.learning_rate, beta1=0.9, beta2=0.999)
        optimizer_2 = tf.train.AdamOptimizer(learning_rate=opt.learning_rate)

        op_gcn = optimizer_1.minimize(model.gcnloss)
        op_bcn = optimizer_2.minimize(model.multiloss)

        sess.run(tf.global_variables_initializer())
        best_acc = 0
        best_f1 = 0
        best_epoch = 0
        best_acc_per_class = 0

        val_feed_dict = train_feed_dict.copy()
        val_feed_dict[placeholders['dropout']] = 0
        IPython.embed()
        for epoch in range(opt.epochs):
            gcn_loss, _ = sess.run([model.gcnloss, op_gcn], feed_dict=train_feed_dict)
            bcn_loss, _ = sess.run([model.multiloss, op_bcn], feed_dict=train_feed_dict)
            val_pred = sess.run(model.basic_logits, feed_dict=val_feed_dict)
            pred_labels = np.argmax(val_pred, axis=1)
            acc = accuracy_score(true_class, pred_labels)
            f1 = f1_score(true_class, pred_labels, average='macro')
            acc_per_class = accuracy_per_class(y_pred=pred_labels, y_true=true_class)
            print('[*] Epoch:', '%d' % (epoch + 1),
                  'gcn_loss=', '{:.5f}'.format(gcn_loss),
                  'bcn_loss=', '{:.5f}'.format(bcn_loss),
                  'val_acc=', '{:.5f}'.format(acc),
                  'val_f1=', '{:.5f}'.format(f1))
            if f1 > best_f1:
                best_acc = acc
                best_f1 = f1
                best_epoch = epoch
                best_acc_per_class = acc_per_class
        print('[Best] Epoch:', '%d' % best_epoch,
                  'val_acc=', '{:.5f}'.format(best_acc),
                  'val_f1=', '{:.5f}'.format(best_f1))
        print('acc_per_class= ', best_acc_per_class)
    elif opt.train_method == 'sequential':
        if opt.matrix_completion == 'sparse':
            responses_complete = responses
        if opt.matrix_completion == 'full':
            res_train = rating_mx_train - 1
            res_train = res_train.T
            for i in range(res_train.shape[0]):
                counts = np.bincount(res_train[i][res_train[i] != -1])
                most = np.argmax(counts)
                res_train[i][res_train[i] == -1] = most
            responses_complete = transform_onehot(res_train, num_oracles, len(class_values))
            IPython.embed()
        elif opt.matrix_completion == 'gcn':
            threshold = 0
            mx, responses_complete = matrix_completion_GCN(u_features, v_features, rating_mx_train, train_labels, train_u_indices,
                         train_v_indices, class_values, threshold)
            # nonres_mask = np.equal(responses, -1)
            # responses_complete[np.logical_not(nonres_mask)] = responses[np.logical_not(nonres_mask)]

            IPython.embed()
        elif opt.matrix_completion == 'nmf':
            matrix_pred = matrix_completion_NMF(rating_mx_train)
            responses_complete = transform_onehot(matrix_pred.T, num_oracles, num_classes)
        if opt.label_aggregation == 'iwmv':
            # sparse + iwmv
            predictY, weight, id_conf = IWMV(responses_complete, class_values, dict())
            acc_iwmv_trn = accuracy_score(predictY, true_class)
            acc_iwmv_per_cls = accuracy_per_class(true_class, predictY)
            f1_iwmv = f1_score(true_class, predictY, average='macro')
            accuracy_trn = (acc_iwmv_trn, f1_iwmv, acc_iwmv_per_cls)
        elif opt.label_aggregation == 'crowd':
            model = build_base_model(dim_instance, num_classes)
            model.add(CrowdsClassification(num_classes, num_oracles, conn_type="MW"))  # sparse + crowd layer
            # instantiate specialized masked loss to handle missing answers
            loss = multi_loss
            model.compile(optimizer='adam', loss=loss)
            model.fit(v_features.toarray(), responses_complete, epochs=opt.epoch_bcn, shuffle=True,
                            batch_size=BATCH_SIZE, verbose=2)
            model.pop()
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            accuracy_trn = eval_model(model, v_features, true_class)
            # accuracy_tst = eval_model(model, v_features_tst, true_class_tst)
        print('Matrix Completion: %s, Label Aggregation: %s' % (opt.matrix_completion, opt.label_aggregation))
        print('Train acc/F1/acc per cls: ', accuracy_trn)
        print('Test acc/F1/acc per cls: ', accuracy_tst)
        # IPython.embed()


def multi_loss(y_true, y_pred):
    vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1)
    mask = tf.equal(y_true[:,0,:], -1)
    zer = tf.zeros_like(vec)
    loss = tf.where(mask, x=zer, y=vec)
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(parser)
    opt = parser.parse_args()
    train(opt)

