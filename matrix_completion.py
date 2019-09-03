from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score
import numpy as np

'''
    NOTICE: only used in binary case!
    input: a sparse matrix, csr_matrix, |oracle| x |instance|, elem = 1,2,3,..(label + 1), missing = 0
    output: nparray, |instance| x |class| x |oracle|
'''
def binary_NMF(matrix_sparse):

    model = NMF(n_components=10, init='random', random_state=0)
    W = model.fit_transform(matrix_sparse)
    H = model.components_
    matrix_pred = W.dot(H)

    for i in range(matrix_pred.shape[0]):
        for j in range(matrix_pred.shape[1]):
            if abs(matrix_pred[i][j] - 1) < abs(matrix_pred[i][j] - 2):
                matrix_pred[i][j] = 1
            else:
                matrix_pred[i][j] = 2
    matrix_pred = matrix_pred.astype(int)
    return matrix_pred


def matrix_completion_NMF(rating_mx_train, train_labels, train_u_indices, train_v_indices, class_values, trn_instance_idx):
    '''
    :param rating_mx_train: csr_matrix, M x N,
    :param train_labels:
    :param train_u_indices:
    :param train_v_indices:
    :param class_values: sorted unique class. eg:[0., 1.]
    :param trn_instance_idx:
    :return:
    '''
    matrix_pred = binary_NMF(rating_mx_train)  # |user| x |item|, elem = {1,2}
    observed_pred = np.array([matrix_pred[train_u_indices[i]][train_v_indices[i]] - 1 for i in range(len(train_u_indices))])
    acc = accuracy_score(train_labels, observed_pred)
    print("nmf acc: ", acc)

    num_items = rating_mx_train.shape[1]
    num_oracle = rating_mx_train.shape[0]
    response_all = np.full((num_items, num_oracle, len(class_values)), -1)
    for oi in range(matrix_pred.shape[0]):
        for id in range(matrix_pred.shape[1]):
            response_all[id][oi] = 0
            response_all[id][oi][matrix_pred[oi][id]-1] = 1

    response_all = response_all.swapaxes(1, 2)
    response_all = response_all[trn_instance_idx]
    return response_all

