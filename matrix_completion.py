from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score
import numpy as np
import IPython
'''
    NOTICE: only used in binary case!
    input: a sparse matrix, csr_matrix, |oracle| x |instance|, elem = 1,2,3,..(label + 1), missing = 0
    output: nparray, |instance| x |class| x |oracle|
'''

def binary_NMF(matrix_sparse):
    model = NMF(n_components=10, init='random', random_state=0, solver='mu')
    matrix_sparse = matrix_sparse.astype(np.float32)
    matrix_sparse[matrix_sparse == 0] = np.nan
    W = model.fit_transform(matrix_sparse)
    H = model.components_
    matrix_pred = W.dot(H)

    '''
    for i in range(matrix_pred.shape[0]):
        for j in range(matrix_pred.shape[1]):
            if abs(matrix_pred[i][j] - 1) < abs(matrix_pred[i][j] - 2):
                matrix_pred[i][j] = 1
            else:
                matrix_pred[i][j] = 2
    matrix_pred = matrix_pred.astype(int)
    '''
    return matrix_pred


def compute_rmse(true, pred):
    err = true - pred
    mse = np.mean(np.square(err))
    return np.sqrt(mse)


def matrix_completion_NMF(rating_mx_train):
    '''
    :param rating_mx_train: csr_matrix, M x N,
    :param train_labels:
    :param train_u_indices:
    :param train_v_indices:
    :param class_values: sorted unique class. eg:[0., 1.]
    :param trn_instance_idx:
    :return:
    '''
    # rating_mx_train = rating_mx_train.toarray()
    matrix_pred = binary_NMF(rating_mx_train)  # |user| x |item|, elem = {1,2}

    # true = response_mx[test_u_indices, test_v_indices]
    # pred = matrix_pred[test_u_indices, test_v_indices]
    # mse = compute_rmse(true, pred)
    # print('nmf rmse: ', rmse)

    '''
    pred_label = []
    for v in pred:
        if v == 0:
            pred_label.append(v)
            continue
        if abs(1-v) < abs(2-v):
            pred_label.append(1)
        else:
            pred_label.append(2)
    acc = accuracy_score(true, pred_label)
    print("nmf acc: ", acc)
    '''

    for i in range(matrix_pred.shape[0]):
        for j in range(matrix_pred.shape[1]):
            if abs(matrix_pred[i][j] - 1) < abs(matrix_pred[i][j] - 2):
                matrix_pred[i][j] = 0
            else:
                matrix_pred[i][j] = 1
    matrix_pred = matrix_pred.astype(int)
    return matrix_pred

