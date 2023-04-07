'''
Tensorflow Implementation of r-Adjnorm model in:
Minghao Zhao  et al. Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering. In SIGIR 2022.

@author: Minghao Zhao(zhaominghao@corp.netease.com)
'''
####################################################
# This section of code adapted from WangXiang/NGCF
# adding  dropedge and PPNW
###################################################
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from scipy.sparse.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from utility.parser import parse_args

args = parse_args()

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        val_file = path + '/val.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.n_val = 0
        self.neg_pools = {}

        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)

        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_val += len(items)



        self.n_items += 1
        self.n_users += 1

        #self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        self.val_set = {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items


        with open(val_file) as f_val:
            for l in f_val.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')]
                except Exception:
                    continue

                uid, val_items = items[0], items[1:]
                self.val_set[uid] = val_items
        
        train_items_list = []
        for each in self.train_items.keys():
            train_items_list += self.train_items[each]
        from collections import Counter
        self.item_cnt = Counter(train_items_list)
        self.sample_list = []
        for i  in range(self.n_items):
            self.sample_list += [i] * int(self.item_cnt[i]**args.ns)


    def get_adj_mat(self):

        adj_mat, norm_adj_mat, mean_adj_mat, item_pop_rev, d_adj = self.create_adj_mat()
 
        return adj_mat, norm_adj_mat, mean_adj_mat, item_pop_rev, d_adj

  
    def get_sample_adj_mat(self, percent, pop_penalty):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)        
        adj_mat = adj_mat.tolil()
        R = self.R.tocoo()#self.R.tolil()
        def normalized_adj_single(adj, bi=False):
            if not bi:
                rowsum = np.array(adj.sum(1))
                rowsum[rowsum==0.] = np.inf
                d_inv = np.power(rowsum, -0.5).flatten()
                d_mat_inv = sp.diags(d_inv)
                norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
                return norm_adj.tocoo()
            else:
                rowsum = np.array(adj.sum(1))
                colsum = np.array(adj.sum(0))
                rowsum[rowsum==0.] = np.inf
                colsum[colsum==0.] = np.inf
                d_inv = np.power(rowsum, -0.).flatten()
                d_mat_inv = sp.diags(d_inv)
                d_inv_ = np.power(colsum, -1.).flatten()
                d_mat_inv_ = sp.diags(d_inv_)                
                norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv_)
                return norm_adj.tocoo()
            
        def randomedge_sampler(train_adj, percent):
            """
            Randomly drop edge and preserve percent% edges.
            """
            "Opt here"       
            nnz = train_adj.nnz
            #perm = np.random.permutation(nnz)
            preserve_nnz = int(nnz*percent)
            if pop_penalty:
               weights = normalized_adj_single(train_adj, bi=True).data
               norm_weights = weights / weights.sum()
               nnz = len(norm_weights)
               preserve_nnz = int(nnz*percent)
               perm = np.random.choice(nnz, preserve_nnz, replace=False, p=norm_weights)
            else:
               perm = np.random.permutation(nnz)
            perm = perm[:preserve_nnz]
            r_adj = sp.coo_matrix((train_adj.data[perm],
                                   (train_adj.row[perm],
                                    train_adj.col[perm])),
                                  shape=train_adj.shape)
            return r_adj
        R = randomedge_sampler(R, percent)
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T        
        adj_mat = adj_mat.todok()
        mean_adj_mat = normalized_adj_single(adj_mat)
        return mean_adj_mat 


    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        print('n_interactions', len(R.nonzero()[0]))
        

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        item_pop = np.array(adj_mat.sum(1)).flatten()[self.n_users:]
        item_pop_rev = item_pop#1 / (item_pop+1)
        t2 = time()

        def normalized_adj_single(adj, verbose= False):
            rowsum = np.array(adj.sum(1))

            r =  args.r
            d_inv = np.power(rowsum, r-1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            d_inv_ = np.power(rowsum, -r).flatten()
            d_inv_[np.isinf(d_inv_)] = 0.
            d_mat_inv_ = sp.diags(d_inv_)

            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv_)
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat, verbose=True)
        print('already normalize adjacency matrix', time() - t2)
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -1.).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr(), item_pop_rev, d_mat_inv.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                if args.positive_sample == 1:

                    while True:
                        tmp = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                        if np.random.rand() > 1- np.sqrt(1/self.item_cnt[pos_items[tmp]]):
                            pos_i_id = pos_items[tmp]
                            break
                else:
                    pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                    pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                if args.negative_sample == 1:
                    neg_id = self.sample_list[np.random.randint(low=0, high=len(self.sample_list), size=1)[0]]                  
                    assert neg_id <= self.n_items
                else:
                    neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def choice(probs):
            x = np.random.rand()
            cum = 0
            for i,p in enumerate(probs):
                cum += p
                if x < cum:
                   break
            return i


        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        ('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_val=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_val, self.n_test, (self.n_train + self.n_val + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state


    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state

    def cosin(self):
        feature_vectors = self.R.T
        similarities = cosine_similarity(feature_vectors)
        for i in range(similarities.shape[0]):
            similarities[i,i] = 1.
        return similarities

    def get_ppnw(self):
        ui_mat = self.R.tocoo()
        colsum = np.array(ui_mat.sum(0)).flatten()
        colsum[colsum==0.]=1#divide 0 error
        rowsum = np.array(ui_mat.sum(1)).flatten()
        rowsum[rowsum==0.]=1#divide 0 error
        all_sum = ui_mat.sum()
        theta_i = np.log(all_sum) -  np.log(colsum)
        d_inv = np.power(colsum, -1)
        d_inv[np.isinf(d_inv)] = 0.
        d_inv_ = np.power(rowsum, -1)
        d_inv_[np.isinf(d_inv_)] = 0.
        theta_i_mat = all_sum * ui_mat.dot(sp.diags(d_inv))
        theta_i_mat.data -= 1
        theta_i_mat = theta_i_mat.log1p()
        theta_u = np.array((sp.diags(d_inv_).dot(theta_i_mat)).sum(1)).flatten()
        theta_i_mat.data -= theta_u[np.nonzero(theta_i_mat)[0]]
        theta_i_mat.data = np.power(theta_i_mat.data, 2)
        theta_std_u2 =  np.array((sp.diags(d_inv_).dot(theta_i_mat)).sum(1)).flatten()

        theta_i_z = (theta_i - np.min(theta_i)) / (np.max(theta_i)-np.min(theta_i))
        theta_i_p = np.power(theta_i_z, args.ppnw_a)
        return theta_u, theta_i, theta_std_u2, theta_i_p
