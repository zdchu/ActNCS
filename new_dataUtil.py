from gcnetwork.gcmc.preprocessing import *
import random
from gcnetwork.gcmc.utils import *
import IPython

def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets

def transform_onehot(answers, N_ANNOT, N_CLASSES):
    answers_bin_missings = []
    for i in range(len(answers)):
        row = []
        for r in range(N_ANNOT):
            if answers[i, r] == -1:
                row.append(-1 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answers[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)
    return answers_bin_missings


def load_data_dec(DATASET, FEATURES, DATASEED=1234, TESTING=False, datasplit_dir=None, SPLITFROMFILE=False, VERBOSE=True, fold=1, test_rate=0.2):
    random.seed(DATASEED)

    if FEATURES:
        datasplit_path = datasplit_dir + DATASET + str(test_rate) + '.pickle'
    else:
        datasplit_path = datasplit_dir + DATASET + str(test_rate) + '_nofeat.pickle'

    if os.path.isfile(datasplit_path):
        print('Reading dataset splits from file...')
        with open(datasplit_path, 'rb') as f:
            num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features, true_class, \
                responses, rating_mx_train, train_labels, u_train_idx, v_train_idx, test_labels, u_test_idx, v_test_idx, \
        val_labels, u_val_idx, v_val_idx = pkl.load(f)

        if VERBOSE:
            print('Number of users = %d' % num_users)
            print('Number of items = %d' % num_items)
            print('Number of links = %d' % ratings.shape[0])
            print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    else:
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features, true_class, \
        responses, rating_mx_train, train_labels, u_train_idx, v_train_idx, test_labels, u_test_idx, v_test_idx, \
        val_labels, u_val_idx, v_val_idx \
            = create_crowd_data(DATASET, datasplit_dir, seed=DATASEED, verbose=VERBOSE, test_rate=test_rate)

        with open(datasplit_path, 'wb') as f:
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features, true_class, \
                      responses, rating_mx_train, train_labels, u_train_idx, v_train_idx, test_labels, u_test_idx, v_test_idx, \
             val_labels, u_val_idx, v_val_idx], f)

    class_values = np.sort(np.unique(ratings))
    return u_features, v_features, responses, class_values, \
        true_class, rating_mx_train, train_labels, u_train_idx, v_train_idx, test_labels, u_test_idx, v_test_idx, \
        val_labels, u_val_idx, v_val_idx


def create_crowd_data(dataset, data_dir, seed=1234, verbose=True, test_rate=0.2):
    sep = r','
    files = ['response.csv', 'instance.csv', 'oracle.csv', 'label.csv']

    if dataset[:3] == "dec":
        data_dir = data_dir + dataset[4:] + '_'
    if dataset[:6] == 'sensor':
        data_dir = data_dir + dataset[7:] + '_'
    # Load response
    filename = data_dir + files[0]
    dtypes = {
        'oracle_id': np.int64, 'instance_id': np.int64,
        'response': np.float32}

    # use engine='python' to ignore warning about switching to python backend when using regexp for sep
    data = pd.read_csv(filename, sep=sep, header=None,
                       names=['oracle_id', 'instance_id', 'response'], converters=dtypes, engine='python')

    # shuffle here like cf-nade paper with python's own random class
    # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
    data_array = data.as_matrix().tolist()
    random.seed(seed)
    random.shuffle(data_array)
    data_array = np.array(data_array)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['oracle_id'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['instance_id'])
    ratings = data_array[:, 2].astype(dtypes['response'])

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
    ratings = ratings.astype(np.float32)



    # Load instance features
    movies_file = data_dir + files[1]
    movies_df = pd.read_csv(movies_file, sep=sep, header=None, engine='python')
    movies_array = movies_df.as_matrix().astype(np.float32)
    v_features = movies_array[:, 1:]

    # Load user features
    users_file = data_dir + files[2]
    user_df = pd.read_csv(users_file, sep=sep, header=None, engine='python')
    user_array = user_df.as_matrix()
    u_features = user_array[:, 1:]

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    # Load label
    label_file = data_dir + files[3]
    label_df = pd.read_csv(label_file, sep=sep, header=None, engine='python')
    label_array = label_df.as_matrix()
    label_array = label_array[:, 1:]
    # IPython.embed()

    # type
    u_features = u_features.astype(np.float32)
    v_features = v_features.astype(np.float32)
    label_array = label_array.astype(np.int16)
    label_array = label_array.reshape(-1)


    neutral_rating = -1  # sparse

    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())} # orin rating: rating idx

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes_ratings, v_nodes_ratings] = np.array([rating_dict[r] for r in ratings])
    responses = labels.copy()
    labels = labels.reshape([-1])  # no.user x no.item, true response, missing value with -1

    '''
    u_features = []
    for u_res in responses:
        try:
            u_features.append([np.count_nonzero(u_res == 0) / (np.count_nonzero(u_res == 0) + np.count_nonzero(u_res==1)),
                               np.count_nonzero(u_res == 1) / (np.count_nonzero(u_res == 0) + np.count_nonzero(u_res==1))])
        except ZeroDivisionError:
            u_features.append([0])
    u_features = sp.csr_matrix(u_features)
    u_features = u_features.astype(np.float32)
    '''
    # split the dataset into train/ val/ test
    # shuffle the instances
    trn_rate = 0.8
    val_rate = test_rate
    tst_rate = test_rate
    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes_ratings, v_nodes_ratings)])  # observed ratings (u,v)
    all_pair_idx = [i for i in range(pairs_nonzero.shape[0])]
    rating_mx_train = responses + 1.
    # IPython.embed()
    for k in rating_dict:
        ratings[ratings == k] = rating_dict[k]
        label_array[label_array == k] = rating_dict[k]

    if test_rate != 0:
        test_pair_idx = random.sample(all_pair_idx, int(tst_rate * len(pairs_nonzero)))
        remained = [i for i in all_pair_idx if i not in test_pair_idx]
        val_pair_idx = random.sample(remained, int(val_rate * len(pairs_nonzero)))
        trn_pair_idx = [i for i in all_pair_idx if i not in val_pair_idx and i not in test_pair_idx]
        # for gcn
        not_trn_idx = np.hstack((val_pair_idx, test_pair_idx))
        rating_mx_train[pairs_nonzero[not_trn_idx][:, 0], pairs_nonzero[not_trn_idx][:, 1]] = 0.
        rating_mx_train = sp.csr_matrix(rating_mx_train)

        train_pairs_idx = pairs_nonzero[trn_pair_idx]
        u_train_idx, v_train_idx = train_pairs_idx.transpose()
        test_pairs_idx = pairs_nonzero[test_pair_idx]
        u_test_idx, v_test_idx = test_pairs_idx.transpose()
        val_pairs_idx = pairs_nonzero[val_pair_idx]
        u_val_idx, v_val_idx = val_pairs_idx.transpose()

        train_idx = np.array([u * num_items + v for u, v in train_pairs_idx])
        train_labels = labels[train_idx]
        val_idx = np.array([u * num_items + v for u, v in val_pairs_idx])
        val_labels = labels[val_idx]
        test_idx = np.array([u * num_items + v for u, v in test_pairs_idx])
        test_labels = labels[test_idx]
    else:
        trn_pair_idx = [i for i in all_pair_idx]
        np.random.shuffle(trn_pair_idx)
        train_pairs_idx = pairs_nonzero[trn_pair_idx]

        u_train_idx, v_train_idx = train_pairs_idx.transpose()
        u_test_idx, v_test_idx = [], []
        u_val_idx, v_val_idx = [], []

        train_idx = np.array([u * num_items + v for u, v in train_pairs_idx])
        train_labels = labels[train_idx]
        val_labels = []
        test_labels = []

    if verbose:
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features, label_array, \
            responses, rating_mx_train, train_labels, u_train_idx, v_train_idx, test_labels, u_test_idx, v_test_idx, \
        val_labels, u_val_idx, v_val_idx


def loadData(DATASET, FEATURES, DATASEED, TESTING, datasplit_path, SPLITFROMFILE, VERBOSE, fold=1, test_rate=0.2):
    u_features, v_features, responses, class_values, \
        true_class, rating_mx_train, train_labels, u_train_idx, v_train_idx, test_labels, u_test_idx, v_test_idx, \
        val_labels, u_val_idx, v_val_idx\
            = load_data_dec(DATASET, FEATURES, DATASEED, TESTING, datasplit_path, SPLITFROMFILE, VERBOSE, fold, test_rate)


    u_features = np.zeros((u_features.shape[0], len(class_values)))
    for i in range(u_features.shape[0]):
        res = responses[i]
        cont = np.bincount(res[res != -1])
        for j in class_values:
            try:
                u_features[i][int(j)] = cont[int(j)] / np.sum(cont)
            except:
                pass


    responses_mx = responses.copy() + 1
    responses = transform_onehot(np.transpose(responses), u_features.shape[0], len(class_values))
    u_features = sp.csr_matrix(u_features)

    return u_features, v_features, responses, responses_mx, class_values, \
        true_class, rating_mx_train, train_labels, u_train_idx, v_train_idx, test_labels, u_test_idx, v_test_idx, \
        val_labels, u_val_idx, v_val_idx


def read_npy(DATASET, test_rate=0.1):
    # np.random.seed(1234)
    true_class = np.load(DATASET + r'/labels_train.npy')
    v_features = np.load(DATASET + r'/data_train_vgg16.npy')
    responses = np.load(DATASET + r'/answers.npy')

    # index = true_class != 7

    # responses = responses[index]
    v_features = v_features.astype(np.float32)
    # true_class = true_class.astype(np.float32)[index]

    # IPython.embed()

    num_items, num_users = responses.shape
    train_num =  int(num_items * (1 - test_rate))

    responses = responses[:train_num]

    u_features = np.ones((num_users, 1))

    class_values = np.sort(np.unique(true_class))
    # user feature

    u_features = np.zeros((num_users, len(class_values)))
    for i in range(num_users):
        res = responses[:, i]
        cont = np.bincount(res[res != -1])
        for j in class_values:
            try:
                u_features[i][j] = cont[j] / np.sum(cont)
            except:
                pass

    rating_mx_train = responses + 1

    train_labels = []
    u_train_idx = []
    v_train_idx = []

    for v_idx, v_res in enumerate(responses):
        for u_idx, res in enumerate(v_res):
            if res != -1:
                train_labels.append(res)
                u_train_idx.append(u_idx)
                v_train_idx.append(v_idx)

    v_features = np.mean(v_features, (1, 2))
    v_features_trn = v_features[:train_num]

    '''
    v_res_distribution = np.zeros((train_num, len(class_values)))
    for i in range(train_num):
        res = responses[i]
        cont = np.bincount(res[res != -1])
        for j in class_values:
            try:
                v_res_distribution[i][j] = cont[j] / np.sum(cont)
            except:
                pass
    v_features_trn = np.concatenate((v_features_trn, v_res_distribution), 1)
    '''

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)
    v_features_trn = sp.csr_matrix(v_features_trn)
    responses = transform_onehot(responses, num_users, len(class_values))

    v_features_tst = v_features[train_num:]

    true_class_trn = true_class[:train_num]
    true_class_tst = true_class[train_num:]
    return u_features, v_features_trn, responses, class_values, \
        true_class_trn, rating_mx_train.T, train_labels, u_train_idx, v_train_idx, \
           v_features_tst, true_class_tst
