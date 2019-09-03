'''
data loading and preprocessing
'''
from gcnetwork.gcmc.preprocessing import *
import random
import IPython

'''
load data
Input:
    -oracle.csv: oracle side feature: in each row
        oracleid, feat1, feat2, ...
        (oracleid does not need to be continuous)
    -instance.csv: instance side feature:
        instanceid (does not need to be continuous), feat1, feat2, ...
    -response.csv: sparse response matrix:
        oracleid, instanceid, label
    -label.csv:
        instanceid, label
output:
    sorted and renumbered oracle/instance/label
    u_features: csr_matrix, oracle features, size = [M, dim(u)]
    v_features: csr_matrix, instance features, size = [N, dim(v)]
    rating_mx_train: csr_matrix, sparse rating_matrix_train, size = [M, N], elem = response inner idx + 1: 1, 2, 3,...
    train_labels: np array, train dataset response, size = |train rating|, elem = response inner idx: 0, 1, 2,...
    train_u_indices: np array, u_train_idx, size = |train rating|, elem = inner user idx
    v_train_idx: np array, v_train_idx, size = |train rating|, elem = inner item idx
    class_values: np array, unique orin rating [0., 1., ...]
    true_class: np array, true class (not response!) of instances
    responses: no.user x no.item, true response idx, missing value with -1
'''
def loadData(DATASET, FEATURES, DATASEED, TESTING, datasplit_path, SPLITFROMFILE, VERBOSE, fold=1):

    if DATASET[:3] == 'dec':
        '''
        u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values, true_class = \
        load_data_dec(DATASET, FEATURES, DATASEED, TESTING, datasplit_path, SPLITFROMFILE, VERBOSE, fold)
        '''
        u_features, v_features, responses, trn_instance_idx, val_instance_idx, test_instance_idx, class_values, \
        true_class, rating_mx_train, train_labels, u_train_idx, v_train_idx \
            = load_data_dec(DATASET, FEATURES, DATASEED, TESTING, datasplit_path, SPLITFROMFILE, VERBOSE, fold)

        responses = transform_onehot(np.transpose(responses), u_features.shape[0], len(class_values))

    #return u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
    #           val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices, test_v_indices, \
    #           class_values, true_class
    return u_features, v_features, responses, trn_instance_idx, val_instance_idx, test_instance_idx, class_values, \
        true_class, rating_mx_train, train_labels, u_train_idx, v_train_idx

def load_data_dec(DATASET, FEATURES, DATASEED=1234, TESTING=False, datasplit_dir=None, SPLITFROMFILE=False, VERBOSE=True, fold=1):
    random.seed(DATASEED)

    if FEATURES:
        datasplit_path = datasplit_dir + DATASET + '.pickle'
    else:
        datasplit_path = datasplit_dir + DATASET + '_nofeat.pickle'

    if os.path.isfile(datasplit_path):
        print('Reading dataset splits from file...')
        with open(datasplit_path) as f:
            num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features, true_class = pkl.load(f)

        if VERBOSE:
            print('Number of users = %d' % num_users)
            print('Number of items = %d' % num_items)
            print('Number of links = %d' % ratings.shape[0])
            print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    else:
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features, true_class \
            = create_crowd_data(DATASET, datasplit_dir, seed=DATASEED, verbose=VERBOSE)

        with open(datasplit_path, 'w') as f:
            IPython.embed()
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features, true_class], f)

    neutral_rating = -1  # sparse

    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist()) } # orin rating: rating idx

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    responses = labels.copy()
    labels = labels.reshape([-1])  # no.user x no.item, true response, missing value with -1

    # split the dataset into train/ val/ test
    # shuffle the instances
    trn_rate = 0.8
    val_rate = 0.1
    tst_rate = 0.1
    all_instance_idx = [i for i in range(num_items)]
    test_instance_idx = random.sample(all_instance_idx, int(tst_rate * num_items))
    remained = [i for i in all_instance_idx if i not in test_instance_idx]
    val_instance_idx = random.sample(remained, int(val_rate * num_items))
    trn_instance_idx = [i for i in all_instance_idx if i not in val_instance_idx and i not in test_instance_idx]

    # for gcn
    rating_mx_train = responses + 1.
    not_trn_idx = np.hstack((val_instance_idx, test_instance_idx))
    rating_mx_train[:, not_trn_idx] = 0.
    rating_mx_train = sp.csr_matrix(rating_mx_train)

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])  # observed ratings (u,v)
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])
    train_id_dict = {i: 1 for i in trn_instance_idx}
    pairs_trn_id = [i for i in range(len(pairs_nonzero)) if pairs_nonzero[i][1] in train_id_dict]  # train
    train_pairs_idx = pairs_nonzero[pairs_trn_id]
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    train_idx = idx_nonzero[pairs_trn_id]
    train_labels = labels[train_idx]


    class_values = np.sort(np.unique(ratings))
    return u_features, v_features, responses, trn_instance_idx, val_instance_idx, test_instance_idx, class_values, \
        true_class, rating_mx_train, train_labels, u_train_idx, v_train_idx


def create_crowd_data(dataset, data_dir, seed=1234, verbose=True):
    sep = r','
    files = ['response.csv', 'instance.csv', 'oracle.csv', 'label.csv']

    if dataset[:3] == "dec":
        data_dir = data_dir + dataset[4:] + '_'

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

    # type
    u_features = u_features.astype(np.float32)
    v_features = v_features.astype(np.float32)
    label_array = label_array.astype(np.int16)
    label_array = label_array.reshape(-1)

    if verbose:
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features, label_array


'''
input: answers: shape = no.instance x np.oracle, missing = -1, elem = 0,1,2,...
output: shape = no.instance x np.class x no.oracle, one hot, missing = -1
'''
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

'''
input: response array, elem = 0,1,...
output: one-hot array, elem = [one hot]
'''
def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets
