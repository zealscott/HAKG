import math
import random
import torch
import numpy as np
from tqdm import tqdm
from time import time
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.data_loader import load_data
from modules.HAKG import Recommender
from utils.evaluate import test
from utils.helper import early_stopping

n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

def get_feed_data(train_entity_pairs, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = list()
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            each_negs = list()
            neg_item = np.random.randint(low=0, high=n_items, size=args.num_neg_sample)
            if len(set(neg_item) & set(train_user_set[user]))==0:
                each_negs += list(neg_item)
            else:
                neg_item = list(set(neg_item) - set(train_user_set[user]))
                each_negs += neg_item
                while len(each_negs)<args.num_neg_sample:
                    n1 = np.random.randint(low=0, high=n_items, size=1)[0]
                    if n1 not in train_user_set[user]:
                        each_negs += [n1]
            neg_items.append(each_negs)

        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,train_user_set))
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best = 0
    stopping_step = 0
    should_stop = False

    print("start training ...")
    iter = math.ceil(len(train_cf_pairs) / args.batch_size)
    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        if epoch%20 == 1 or epoch==0:
            # shuffle training data
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_pairs = train_cf_pairs[index]
            print("start prepare feed data...")
            all_feed_data = get_feed_data(train_cf_pairs, user_dict['train_user_set'])  # {'user': [n,], 'pos_item': [n,], 'neg_item': [n, n_sample]}

        """training"""
        model.train()
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        for i in tqdm(range(iter)):
            batch = dict()
            batch['users'] = all_feed_data['users'][i*args.batch_size:(i+1)*args.batch_size].to(device)
            batch['pos_items'] = all_feed_data['pos_items'][i*args.batch_size:(i+1)*args.batch_size].to(device)
            batch['neg_items'] = all_feed_data['neg_items'][i*args.batch_size:(i+1)*args.batch_size,:].to(device)

            batch_loss = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            s += args.batch_size
        train_e_t = time()

        if epoch % 5 == 0 or epoch == 1:
            """testing"""
            model.eval()
            test_s_t = time()
            with torch.no_grad():
                ret = test(model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
            )
            print(train_res)
            f = open('./result/{}.txt'.format(args.dataset), 'a+')
            f.write(str(train_res) + '\n')
            f.close()

            # *********************************************************
            cur_best, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=20)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

        else:
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best))
