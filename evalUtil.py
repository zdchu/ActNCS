import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

def build_base_model(input_dim, N_CLASSES):
    base_model = Sequential()
    base_model.add(Dense(1024, input_shape=(input_dim,), activation='relu'))
    base_model.add(Dropout(0.5))
    base_model.add(Dense(256, input_shape=(input_dim,), activation='relu'))
    base_model.add(Dropout(0.5))
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

