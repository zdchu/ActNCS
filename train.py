from __future__ import division
from __future__ import print_function

import argparse
import IPython

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import sys
import json

import dataUtil

import matplotlib.cm as cm
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.extmath import softmax
from sklearn.metrics import accuracy_score

from gcnetwork.gcmc.preprocessing import *
from crowdlayer.conlleval import conlleval

# packages for learning from crowds
from crowdlayer.crowd_layer.crowd_layers import CrowdsClassification, MaskedMultiCrossEntropy
from crowdlayer.crowd_layer.crowd_aggregators import CrowdsCategoricalAggregator

from gcnetwork.gcmc.preprocessing import create_trainvaltest_split, \
    sparse_to_tuple, preprocess_user_item_features, globally_normalize_bipartite_adjacency, \
    load_data_monti, load_official_trainvaltest_split, normalize_features, load_data_sensor, load_data_crowd
from gcnetwork.gcmc.model import RecommenderGAE, RecommenderSideInfoGAE
from gcnetwork.gcmc.utils import construct_feed_dict

from matrix_completion import matrix_completion_NMF
from labelAggregation import IWMV

# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="dec_appendicitis",
                choices=['dec_omega3','dec_appendicitis','dec_dst'],
                help="Dataset string.")

ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                help="Learning rate")

ap.add_argument("-e", "--epochs", type=int, default=200,
                help="Number training epochs")

ap.add_argument("-hi", "--hidden", type=int, nargs=2, default=[500, 75],
                help="Number hidden units in 1st and 2nd layer")

ap.add_argument("-fhi", "--feat_hidden", type=int, default=64,
                help="Number hidden units in the dense layer for features")
ap.add_argument("-ds", "--data_seed", type=int, default=1234,
                help="""Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324).
                     Only used for ml_1m and ml_10m datasets. """)
ap.add_argument("-ac", "--accumulation", type=str, default="sum", choices=['sum', 'stack'],
                help="Accumulation function: sum or stack.")
ap.add_argument("-do", "--dropout", type=float, default=0.7,
                help="Dropout fraction")
ap.add_argument("-nb", "--num_basis_functions", type=int, default=2,
                help="Number of basis functions for Mixture Model GCN.")


# Boolean flags
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-nsym', '--norm_symmetric', dest='norm_symmetric',
                help="Option to turn on symmetric global normalization", action='store_true')
fp.add_argument('-nleft', '--norm_left', dest='norm_symmetric',
                help="Option to turn on left global normalization", action='store_false')
ap.set_defaults(norm_symmetric=True)

# side features
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-f', '--features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_true')
fp.add_argument('-no_f', '--no_features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_false')
ap.set_defaults(features=True)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-t', '--testing', dest='testing',
                help="Option to turn on test set evaluation", action='store_true')
fp.add_argument('-v', '--validation', dest='testing',
                help="Option to only use validation set evaluation", action='store_false')
ap.set_defaults(testing=False)


args = vars(ap.parse_args())

print('Settings:')
print(args, '\n')

# Define parameters
DATASET = args['dataset']
DATASEED = args['data_seed']
NB_EPOCH = args['epochs']
HIDDEN = args['hidden']
FEATHIDDEN = args['feat_hidden']
LR = args['learning_rate']
FEATURES = args['features']
TESTING = args['testing']
SYM = args['norm_symmetric']
TESTING = args['testing']
ACCUM = args['accumulation']
BASES = args['num_basis_functions']
DO = args['dropout']

SELFCONNECTIONS = False
SPLITFROMFILE = True
VERBOSE = True

'''
GCN-based matrix completion
'''
def matrix_completion_GCN(u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices,class_values,trn_instance_idx):
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
    orin_u_num = u_features.shape[0]
    filtered_oracle = {}
    responses = np.transpose(adj_train.toarray())
    for id in trn_instance_idx:
        for oi in range(len(responses[id])):
            if responses[id][oi] != 0:
                filtered_oracle[oi] = 1
    filtered_oracle = list(filtered_oracle.keys())
    filtered_oracle.sort()
    orin2inner_u = {filtered_oracle[i]: i for i in range(len(filtered_oracle))}
    inner2orin_u = {i: filtered_oracle[i] for i in range(len(filtered_oracle))}
    u_features = u_features[filtered_oracle]

    # instance
    trn_instance_idx.sort()
    orin2inner_v = {trn_instance_idx[i]: i for i in range(len(trn_instance_idx))}
    v_features = v_features[trn_instance_idx]

    adj_train = adj_train[:, trn_instance_idx]
    adj_train = adj_train[filtered_oracle, :]
    train_u_indices = [orin2inner_u[u] for u in train_u_indices]
    train_v_indices = [orin2inner_v[v] for v in train_v_indices]
    #
    num_users, num_items = adj_train.shape
    NUMCLASSES = len(class_values)
    num_side_features = 0

    # feature loading
    if not FEATURES:
        u_features = sp.identity(num_users, format='csr')
        v_features = sp.identity(num_items, format='csr')

        u_features, v_features = preprocess_user_item_features(u_features, v_features)

    elif FEATURES and u_features is not None and v_features is not None:
        # use features as side information and node_id's as node input features

        print("Normalizing feature vectors...")
        u_features_side = normalize_features(u_features)
        v_features_side = normalize_features(v_features)

        u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

        u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
        v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

        num_side_features = u_features_side.shape[1]

        # node id's for node input features
        id_csr_v = sp.identity(num_items, format='csr')
        id_csr_u = sp.identity(num_users, format='csr')

        u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

    else:
        raise ValueError('Features flag is set to true but no features are loaded from dataset')

    # global normalization
    support = []
    support_t = []
    adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)

    for i in range(NUMCLASSES):
        # build individual binary rating matrices (supports) for each rating
        support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)

        if support_unnormalized.nnz == 0:
            # yahoo music has dataset split with not all ratings types present in training set.
            # this produces empty adjacency matrices for these ratings.
            sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

        support_unnormalized_transpose = support_unnormalized.T
        support.append(support_unnormalized)
        support_t.append(support_unnormalized_transpose)

    support = globally_normalize_bipartite_adjacency(support, symmetric=SYM)
    support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=SYM)

    if SELFCONNECTIONS:
        support.append(sp.identity(u_features.shape[0], format='csr'))
        support_t.append(sp.identity(v_features.shape[0], format='csr'))

    num_support = len(support)
    support = sp.hstack(support, format='csr')
    support_t = sp.hstack(support_t, format='csr')

    if ACCUM == 'stack':
        div = HIDDEN[0] // num_support
        if HIDDEN[0] % num_support != 0:
            print("""\nWARNING: HIDDEN[0] (=%d) of stack layer is adjusted to %d such that
                      it can be evenly split in %d splits.\n""" % (HIDDEN[0], num_support * div, num_support))
        HIDDEN[0] = num_support * div

    # Collect all user and item nodes for train set
    train_u = list(set(train_u_indices))
    train_v = list(set(train_v_indices))
    train_u_dict = {n: i for i, n in enumerate(train_u)}
    train_v_dict = {n: i for i, n in enumerate(train_v)}

    train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
    train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])

    train_support = support[np.array(train_u)]
    train_support_t = support_t[np.array(train_v)]

    # features as side info
    if FEATURES:
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
    # create model
    if FEATURES:
        model = RecommenderSideInfoGAE(placeholders,
                                       input_dim=u_features.shape[1],
                                       feat_hidden_dim=FEATHIDDEN,
                                       num_classes=NUMCLASSES,
                                       num_support=num_support,
                                       self_connections=SELFCONNECTIONS,
                                       num_basis_functions=BASES,
                                       hidden=HIDDEN,
                                       num_users=num_users,
                                       num_items=num_items,
                                       accum=ACCUM,
                                       learning_rate=LR,
                                       num_side_features=num_side_features,
                                       logging=True)
    else:
        model = RecommenderGAE(placeholders,
                               input_dim=u_features.shape[1],
                               num_classes=NUMCLASSES,
                               num_support=num_support,
                               self_connections=SELFCONNECTIONS,
                               num_basis_functions=BASES,
                               hidden=HIDDEN,
                               num_users=num_users,
                               num_items=num_items,
                               accum=ACCUM,
                               learning_rate=LR,
                               logging=True)

    train_support = sparse_to_tuple(train_support)
    train_support_t = sparse_to_tuple(train_support_t)

    u_features = sparse_to_tuple(u_features)
    v_features = sparse_to_tuple(v_features)
    assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

    num_features = u_features[2][1]
    u_features_nonzero = u_features[1].shape[0]
    v_features_nonzero = v_features[1].shape[0]

    # Feed_dicts for validation and test set stay constant over different update steps
    train_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                          v_features_nonzero, train_support, train_support_t,
                                          train_labels, train_u_indices, train_v_indices, class_values, DO,
                                          train_u_features_side, train_v_features_side)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('Training GAE...')

    for epoch in range(NB_EPOCH):
        outs = sess.run([model.training_op, model.loss, model.rmse, model.accuracy, model.outputs_all, model.outputs], feed_dict=train_feed_dict)
        train_avg_loss = outs[1]
        train_rmse = outs[2]
        train_acc = outs[3]

        output_all = outs[4]  # (No. user_trn x No. item_trn) x No. class
        output_obs = outs[5]  # |observed response in trn| x No. class
        #acc = accuracy_res_sensor(res_matrix_all, outputs_all, u_part2all, v_part2all)


        if VERBOSE:
            print("[*] Epoch:", '%04d' % (epoch + 1),
                  #"my acc=", "{:.5f}".format(acc),
                  "train acc=", "{:.5f}".format(train_acc),
                  "train_loss=", "{:.5f}".format(train_avg_loss),
                  "train_rmse=", "{:.5f}".format(train_rmse))

    probs = softmax(output_all)
    pred_labels = np.argmax(probs, axis=1)  # |num. oracle_trn x num. instance_trn|
    pred_labels = pred_labels.reshape((num_users, num_items))

    response_all = np.full((num_items, orin_u_num, len(class_values)), -1)
    for oi in range(pred_labels.shape[0]):
        for id in range(pred_labels.shape[1]):
            response_all[id][inner2orin_u[oi]] = 0
            response_all[id][inner2orin_u[oi]][pred_labels[oi][id]] = 1

    response_all = response_all.swapaxes(1, 2)

    return response_all


'''
    input data assumption:
        -the set of response label contains all the labels
'''
def train():
    datasplit_path = "./gcnetwork/gcmc/data/dec/complete/"
    fold = 1
    N_EPOCHS = 300
    BATCH_SIZE = 100

    u_features, v_features, responses, trn_instance_idx, val_instance_idx, test_instance_idx, class_values, \
    true_class, rating_mx_train, train_labels, train_u_indices, train_v_indices \
        = dataUtil.loadData(DATASET, FEATURES, DATASEED, TESTING, datasplit_path, SPLITFROMFILE, VERBOSE, fold)
    #transform
    num_classes = len(class_values)
    num_oracles = u_features.shape[0]
    dim_instance = v_features.shape[1]

    # matrix completion
    # train dataset
    #response_complete = matrix_completion_GCN(u_features, v_features, rating_mx_train, train_labels, train_u_indices,train_v_indices, class_values, trn_instance_idx)
    response_complete = matrix_completion_NMF(rating_mx_train, train_labels, train_u_indices, train_v_indices, class_values, trn_instance_idx)


    # for fold
    x_train = v_features[trn_instance_idx].toarray()
    y_train = true_class[trn_instance_idx]
    x_test = v_features[test_instance_idx].toarray()
    y_test = true_class[test_instance_idx]

    # sparse + iwmv
    predictY, weight, id_conf = IWMV(responses[trn_instance_idx], class_values, dict())
    acc_iwmv_trn = accuracy_score(y_train, predictY)
    acc_iwmv_per_cls = accuracy_per_class(y_train, predictY)
    #print("iwmv train acc/F1/acc per cls: ",acc_iwmv_trn, acc_iwmv_per_cls)
    model0 = build_base_model(dim_instance, num_classes)

    y_imwv_trn = np.zeros((len(x_train),2))
    for i in range(len(predictY)):
        y_imwv_trn[i][int(predictY[i])] = 1
    y_imwv_tst = np.zeros((len(y_test),2))
    for i in range(len(y_test)):
        y_imwv_tst[i][int(y_test[i])] = 1

    model0.fit(x_train, y_imwv_trn, epochs=60)
    accuracy_trn0 = eval_model(model0, x_train, y_train)
    accuracy_tst0 = eval_model(model0, x_test, y_test)
    print('iwmv train acc/F1/acc per cls: ', accuracy_trn0, 'test acc/F1/acc per cls:', accuracy_tst0)

    # complete + iwmv
    predictY, weight, id_conf = IWMV(response_complete, class_values, dict())
    acc_iwmv_trn = accuracy_score(y_train, predictY)
    acc_iwmv_per_cls = accuracy_per_class(y_train, predictY)
    #print("iwmv train acc/F1/acc per cls: ", acc_iwmv_trn, acc_iwmv_per_cls)
    model0 = build_base_model(dim_instance, num_classes)

    y_imwv_trn = np.zeros((len(x_train), 2))
    for i in range(len(predictY)):
        y_imwv_trn[i][int(predictY[i])] = 1
    y_imwv_tst = np.zeros((len(y_test), 2))
    for i in range(len(y_test)):
        y_imwv_tst[i][int(y_test[i])] = 1

    model0.fit(x_train, y_imwv_trn, epochs=60)
    accuracy_trn0 = eval_model(model0, x_train, y_train)
    accuracy_tst0 = eval_model(model0, x_test, y_test)
    print('iwmv-complete train acc/F1/acc per cls: ', accuracy_trn0, 'test acc/F1/acc per cls:', accuracy_tst0)


    # prediction block (bottleneck layer)
    model = build_base_model(dim_instance, num_classes)
    model2 = build_base_model(dim_instance, num_classes)

    # crowd layer
    # add crowds layer on top of the base model
    model.add(CrowdsClassification(num_classes, num_oracles, conn_type="MW"))  # sparse + crowd layer
    model2.add(CrowdsClassification(num_classes, num_oracles, conn_type="MW"))  # complete + crowd layer

    # instantiate specialized masked loss to handle missing answers
    loss = MaskedMultiCrossEntropy().loss
    loss2 = MaskedMultiCrossEntropy().loss

    # compile model with masked loss and train
    model.compile(optimizer='adam', loss=loss)
    model.fit(x_train, responses[trn_instance_idx], epochs=N_EPOCHS, shuffle=True, batch_size=BATCH_SIZE, verbose=2)
    #
    model2.compile(optimizer='adam', loss=loss2)
    model2.fit(x_train, response_complete, epochs=N_EPOCHS, shuffle=True, batch_size=BATCH_SIZE, verbose=2)

    # save weights from crowds layer for later
    #weights = model.layers[4].get_weights()

    # remove crowds layer before making predictions
    model.pop()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model2.pop()
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    accuracy_trn = eval_model(model, x_train, y_train)
    accuracy_test = eval_model(model, x_test, y_test)

    print('Train acc/F1/acc per cls: ', accuracy_trn)
    print('Test acc/F1/acc per cls: ', accuracy_test)

    accuracy_trn = eval_model(model2, x_train, y_train)
    accuracy_test = eval_model(model2, x_test, y_test)

    print('Train2 acc/F1/acc per cls: ', accuracy_trn)
    print('Test2 acc/F1/acc per cls: ', accuracy_test)


def build_base_model(input_dim, N_CLASSES):
    base_model = Sequential()
    base_model.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    base_model.add(Dropout(0.5))
    base_model.add(Dense(N_CLASSES))
    base_model.add(Activation("softmax"))
    base_model.compile(optimizer='adam', loss='categorical_crossentropy')

    return base_model

def eval_model(model, test_data, test_labels):
    # testset accuracy
    preds_test = model.predict(test_data)
    preds_test_num = np.argmax(preds_test, axis=1)

    classes = list(set(test_labels))
    classes.sort()
    acc_per_class = []
    for i in range(len(classes)):
        instance_class = [j for j in range(len(test_labels)) if test_labels[j] == classes[i]]
        acc_i = accuracy_score(test_labels[instance_class], preds_test_num[instance_class])
        acc_per_class.append(acc_i)
    acc = accuracy_score(test_labels, preds_test_num)
    f1 = f1_score(test_labels, preds_test_num, average='macro')
    return acc, f1, acc_per_class

def accuracy_per_class(y_true, y_pred):
    classes = list(set(y_true))
    classes.sort()
    acc_per_class = []
    for i in range(len(classes)):
        instance_class = [j for j in range(len(y_true)) if y_true[j] == classes[i]]
        acc_i = accuracy_score(y_true[instance_class], y_pred[instance_class])
        acc_per_class.append(acc_i)
    return acc_per_class

if __name__ == '__main__':
    train()
