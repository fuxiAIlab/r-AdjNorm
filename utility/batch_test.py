'''
Tensorflow Implementation of r-Adjnorm model in:
Minghao Zhao  et al. Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering. In SIGIR 2022.

@author: Minghao Zhao(zhaominghao@corp.netease.com)
'''
####################################################
# This section of code adapted from WangXiang/NGCF, 
# adding Novelty@K and PRU@K.
###################################################
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq
import numpy as np


cores = multiprocessing.cpu_count()

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


train_items = data_generator.train_items
train_items_list = []
pop_user = np.zeros(USR_NUM)
for each in train_items.keys():
    train_items_list += train_items[each]
    pop_user[each] = len(train_items[each])

degree_rev = []
from collections import Counter

item_cnt = Counter(train_items_list)
item_sum = len(train_items_list)
item_dis = len(set(train_items_list))+1
pop_item = []
for i in range(ITEM_NUM):
    if item_cnt[i] == 0:
        pop_item.append(item_cnt[i]+1)
    else:
        pop_item.append(item_cnt[i])

pop_item = np.array(pop_item)


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, len(user_pos_test)))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    rating = x[0]
    u = x[1]
    is_test_flag = x[2]
    try:
        training_items = data_generator.train_items[u]
        val_items = data_generator.val_set[u]
    except Exception:
        training_items = []
        val_items = []

    all_items = set(range(ITEM_NUM))

    if is_test_flag:
        user_pos_test = data_generator.test_set[u]
        test_items = list(all_items - set(training_items) - set(val_items))
    else:
        user_pos_test = data_generator.val_set[u]
        test_items = list(all_items - set(training_items))


    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(sess, model, adj, users_to_test, topk, flag, is_test, drop_flag=False, batch_test_flag=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    model.norm_adj = adj
    if is_test and topk==20:
        print('test adj nonzero', len(model.norm_adj.nonzero()[0]))
    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2 
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    rate_test = np.array([])

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                else:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.]*len(eval(args.layer_size))})
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                              model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                              model.pos_items: item_batch,
                                                              model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                              model.mess_dropout: [0.] * len(eval(args.layer_size))})
        if is_test:
            test_flag = True
        else:
            test_flag = False
        test_flag_list = [test_flag for i in range(rate_batch.shape[0])]
        user_batch_rating_uid = zip(rate_batch, user_batch, test_flag_list)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        #topk = 10
        if flag:
            for i in range(len(user_batch)):
                for item in data_generator.train_items[user_batch[i]]:
                    rate_batch[i, item] = -np.inf
                for item_ in data_generator.val_set[user_batch[i]]:
                    rate_batch[i, item_] = -np.inf
            if args.pc == 1:
                def norm(user_predict, M, user_item_cnt):
                    user_predict = user_predict.copy()
                    user_predict /= (M - user_item_cnt).reshape(-1, 1)
                    user_predict[user_predict==-np.inf]=0
                    return np.linalg.norm(user_predict, axis=1)
                n = norm(rate_batch, ITEM_NUM, pop_user[user_batch])
                c = 1 / pop_item  * (rate_batch * args.pc_b + 1 - args.pc_b)
                m = norm(c, ITEM_NUM, pop_user[user_batch]) 
                rate_batch +=  args.pc_a * c * (n / m).reshape(-1, 1) 
                if is_test:
                    test_flag = True
                else:
                    test_flag = False
                test_flag_list = [test_flag for i in range(rate_batch.shape[0])]
                user_batch_rating_uid = zip(rate_batch, user_batch, test_flag_list)
                batch_result = pool.map(test_one_user, user_batch_rating_uid)
        
            rate_test = np.append(rate_test, (-rate_batch).argsort()[:,:topk])

        count += len(batch_result)


        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    if flag:
        rate_test = rate_test.reshape(-1, topk)
        novelty = []
        degree = []
        error_cnt = 0
        for i in rate_test:
            for j in i:
                if j in item_cnt:              
                    novelty.append(-np.log2(item_cnt[j] / USR_NUM) / np.log2(USR_NUM))
                    degree.append(item_cnt[j])
                else:
                    error_cnt += 1
       
        cover_ratio = len(set(rate_test.flatten()))/ITEM_NUM
        print('cover_ratio@' + str(topk), cover_ratio)
        print('degree_mean@' + str(topk), np.mean(degree))
        print('novelty@'+ str(topk), np.mean(novelty), 'error cnt', error_cnt)
        
        from scipy import stats
        PRU = []
        for i in rate_test:
            pop = []
            for j in i:
                pop.append(item_cnt[j])
            if sum(np.array(pop)==pop[0])==len(pop):
                pop[-1] += 1e-15
            PRU.append(-stats.spearmanr(range(topk), pop)[0])
        print('PRU@'+ str(topk), np.mean(PRU), 'length', len(PRU))    
       
    
    assert count == n_test_users
    pool.close()

    if not flag:
        return 0, 0, result
    else:
        return  np.mean(novelty), np.mean(PRU), result

