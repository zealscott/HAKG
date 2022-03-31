import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HAKG")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="alibaba-ifashion", help="Choose a dataset:[last-fm,alibaba-ifashion,yelp2018]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='test batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--angle_loss_w', type=float, default=0.005, help='angle loss weight')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    parser.add_argument('--num_neg_sample', type=int, default=200, help='the number of negative sample')
    parser.add_argument('--margin', type=float, default=0.6, help='the margin of contrastive_loss')
    parser.add_argument('--loss_f', nargs="?", default="contrastive_loss",
                        help="Choose a loss function:[inner_bpr, dis_bpr, contrastive_loss]")

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./model_para/", help="output directory for model")

    return parser.parse_args()
