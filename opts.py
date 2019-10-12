from __future__ import print_function

def model_opts(parser):
    parser.add_argument('-m', '--train_method', type=str, default='joint',
                        choices=['joint', 'alternate', 'sequential'],
                        help='Training method')
    parser.add_argument("-d", "--dataset", type=str, default="dec_omega3",
                # choices=['dec_omega3','dec_appendicitis','dec_dst'],
                help="Dataset string")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=200,
                help="Number training epochs")

    parser.add_argument('-eb', '--epoch_bcn', type=int, default=200)

    parser.add_argument("-hi", "--hidden", type=int, nargs=2, default=[500, 100],
                help="Number hidden units in 1st and 2nd layer")
    parser.add_argument("-fhi", "--feat_hidden", type=int, default=128,
                help="Number hidden units in the dense layer for features")
    parser.add_argument("-ds", "--data_seed", type=int, default=1234,
                help="""Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324).
                     Only used for ml_1m and ml_10m datasets. """)
    parser.add_argument("-ac", "--accumulation", type=str, default="sum", choices=['sum', 'stack'],
                help="Accumulation function: sum or stack.")
    parser.add_argument("-do", "--dropout", type=float, default=0.7,
                help="Dropout fraction")
    parser.add_argument("-nb", "--num_basis_functions", type=int, default=4,
                help="Number of basis functions for Mixture Model GCN.")
    parser.add_argument('-mc', '--matrix_completion', type=str, default='sparse',
                choices=['sparse', 'nmf', 'gcn', 'full'],
                help='Method of matrix completion')
    parser.add_argument('-la', '--label_aggregation', type=str, default='iwmv',
                choices=['crowd', 'iwmv'],
                help='Method of label aggregation')
    parser.add_argument('-tr', '--test_rate', type=float, default=0,
                help='proportion of testing data')
    parser.add_argument('-sc', '--self_connections', type=bool, default=False,
                        help='Option to use self connections')
    parser.add_argument('-sp', '--split_from_file', type=bool, default=True,
                        help='Split from file')
    parser.add_argument('-vb', '--verbose', type=bool, default=True,
                        help='Print training details')
    parser.add_argument('-b', '--beta', type=float, default=0.5)
    parser.add_argument('-conn', '--conn_type', type=str, default='MW',
                        choices=['MW', 'VW', 'VB', 'VW+B', 'SW'])
    
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('-nsym', '--norm_symmetric', dest='norm_symmetric',
                help="Option to turn on symmetric global normalization", action='store_true')
    fp.add_argument('-nleft', '--norm_left', dest='norm_symmetric',
                help="Option to turn on left global normalization", action='store_false')
    parser.set_defaults(norm_symmetric=True)

    # side features
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('-f', '--features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_true') 
    fp.add_argument('-no_f', '--no_features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_false')
    parser.set_defaults(features=True)

    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('-t', '--testing', dest='testing',
                help="Option to turn on test set evaluation", action='store_true')
    fp.add_argument('-v', '--validation', dest='testing',
                help="Option to only use validation set evaluation", action='store_false')
    parser.set_defaults(testing=False)
    
