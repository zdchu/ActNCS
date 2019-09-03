import tensorflow as tf
import numpy as np


def softmax_accuracy(preds, labels):
    """
    Accuracy for multiclass model.
    :param preds: predictions
    :param labels: ground truth labelt
    :return: average accuracy
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.to_int64(labels))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def expected_rmse(logits, labels, class_values=None):
    """
    Computes the root mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth label
    :param class_values: rating values corresponding to each class.
    :return: rmse
    """

    probs = tf.nn.softmax(logits)
    if class_values is None:
        scores = tf.to_float(tf.range(start=0, limit=logits.get_shape()[1]) + 1)  # 1 ~ No.class
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        scores = class_values
        y = tf.gather(class_values, labels)

    pred_y = tf.reduce_sum(probs * scores, 1)

    diff = tf.subtract(y, pred_y)
    exp_rmse = tf.square(diff)
    exp_rmse = tf.cast(exp_rmse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(exp_rmse))


def rmse(logits, labels, class_values=None):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    if class_values is None:
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        y = tf.gather(class_values, labels)

    pred_y = logits

    diff = tf.subtract(y, pred_y)
    mse = tf.square(diff)
    mse = tf.cast(mse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(mse))


def softmax_cross_entropy(outputs, labels):
    """ computes average softmax cross entropy """

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
    return tf.reduce_mean(loss)

'''
def accuracy_res_sensor(res_matrix_all, outputs_pred, u_part2all, v_part2all):
    # numpy
    probs = softmax_numpy(outputs_pred)
    pred_labels = np.argmax(probs, axis=1)
    part_v = [v_part2all[v] for v in v_part2all]
    y = res_matrix_all[:, part_v]
    y = y.reshape(-1)
    y = y - 1
    correct = (y == pred_labels).sum()
    size = y.size
    acc = float(correct)/size
    return acc

def accuracy_res_crowd(outputs_pred, train_labels, train_u_indices, train_v_indices):

    # numpy
    probs = softmax_numpy(outputs_pred)
    pred_labels = np.argmax(probs, axis=1)
    part_v = [v_part2all[v] for v in v_part2all]
    y = res_matrix_all[:, part_v]
    y = y.reshape(-1)
    y = y - 1
    correct = (y == pred_labels).sum()
    size = y.size
    acc = float(correct)/size
    return acc

def accuracy_res(y, outputs_pred):
    # numpy
    probs = softmax_numpy(outputs_pred)
    pred_labels = np.argmax(probs, axis=1)

    correct = (y == pred_labels).sum()
    size = y.size
    acc = float(correct) / size
    return acc


def softmax_numpy(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
'''