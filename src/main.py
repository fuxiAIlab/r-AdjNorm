'''
Tensorflow Implementation of r-Adjnorm model in:
Minghao Zhao  et al. Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering. In SIGIR 2022.

@author: Minghao Zhao(zhaominghao@corp.netease.com)
'''
####################################################
# This section of code adapted from WangXiang/NGCF, 
# adding  severl baselines. e.g., pop_reg, PPNW, dropedge
###################################################
import tensorflow as tf
import os
import sys
import csv
import matplotlib.pyplot as plt
from numpy import savetxt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
from utility.batch_test import *


class AdjNorm(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'AdjNorm'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 1

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        self.item_pop_rev = data_config['item_pop_rev']
        self.d = data_config['d']
        self.pop_reg = args.pop_reg
        self.pop_reg_decay = args.pop_reg_decay


        self.ppnw = args.ppnw
        self.ppnw_a = args.ppnw_a
        self.ppnw_g = args.ppnw_g
        self.ppnw_l = args.ppnw_l
        self.theta_u = data_config['theta_u']
        self.theta_i = data_config['theta_i']
        self.theta_i_p = data_config['theta_i_p']
        self.theta_std_u2 = data_config['theta_std_u2']
        '''
        Create Placeholder for Input Data & Dropout.
        '''
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        Create Model Parameters (i.e., Initialize Weights).
        """
        self.weights = self._init_weights()
        """
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
            4. lrgccf: defined in 'Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach',  AAAI2020;
            5. lightgcn: defined in 'LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation', SIGIR2020;
        """
        if self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        elif self.alg_type in ['mf']:
            self.ua_embeddings, self.ia_embeddings = self.weights['user_embedding'], self.weights['item_embedding']

        elif self.alg_type in ['lightgcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
            
        elif self.alg_type in ['lrgccf']:
            self.ua_embeddings, self.ia_embeddings = self._create_lrgccf_embed()

        """
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users) 
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.pos_i_pop_rev = tf.nn.embedding_lookup(self.item_pop_rev, self.pos_items)
        self.neg_i_pop_rev = tf.nn.embedding_lookup(self.item_pop_rev, self.neg_items)
        self.pi_ui = tf.nn.embedding_lookup(self.theta_i_p, self.pos_items) * tf.exp(-tf.math.pow(tf.nn.embedding_lookup(self.theta_u, self.users) - tf.nn.embedding_lookup(self.theta_i, self.pos_items), 2) / (2 * self.ppnw_l * tf.nn.embedding_lookup(self.theta_std_u2, self.users)))
        self.pi_uj = tf.nn.embedding_lookup(self.theta_i_p, self.neg_items) * tf.exp(-tf.math.pow(tf.nn.embedding_lookup(self.theta_u, self.users) - tf.nn.embedding_lookup(self.theta_i, self.neg_items), 2) / (2 * self.ppnw_l * tf.nn.embedding_lookup(self.theta_std_u2, self.users)))
        """
        Inference for the testing phase.
        """
        self.batch_ratings = tf.sigmoid(tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True))

        """
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float64)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float64)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' %k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)


        all_weights['W_att'] = tf.Variable(
            initializer([self.emb_dim, self.emb_dim]), name='W_att')
        all_weights['b_att'] = tf.Variable(
            initializer([1, self.emb_dim]), name='b_att')


        all_weights['s'] = tf.Variable(initializer([self.emb_dim, 1]))
        all_weights['w'] = tf.Variable(initializer([self.n_layers + 1, 1]))
        all_weights['alpha'] = tf.Variable(initializer([1])) 
        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers): 
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append( tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            #if f % 2 == 1:
            all_embeddings += [ego_embeddings]
        if args.single  == 0:
            all_embeddings = tf.stack(all_embeddings, 1)
            all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        else:
            all_embeddings = all_embeddings[-1]

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_lrgccf_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append( tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings

            all_embeddings += [ego_embeddings]
        if args.single  == 0:
            all_embeddings = tf.concat(all_embeddings, axis=1)
        else:
            all_embeddings = all_embeddings[-1]
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


    def _create_ngcf_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.nn.leaky_relu(tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
        if args.single == 1:
            all_embeddings = all_embeddings[-1]
        else:
            all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [embeddings]
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' %k]) + self.weights['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])
            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)       
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = []
        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            mlp_embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k])
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])
            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


    def create_bpr_loss(self, users, pos_items, neg_items):

        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        def pearson_r(y_true, y_pred):
            x = y_true
            y = y_pred
            mx = tf.reduce_mean(x,)
            my = tf.reduce_mean(y,)
            xm, ym = x - mx, y - my
            t1_norm = tf.nn.l2_normalize(xm,)
            t2_norm = tf.nn.l2_normalize(ym,)
            cosine = tf.losses.cosine_distance(t1_norm, t2_norm, axis = -1, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            return 1-cosine
        mf_loss = tf.reduce_mean(tf.math.softplus(-(pos_scores - neg_scores))) #+ 1 * pearson_r(pos_scores, self.pos_i_pop_rev)
        if self.ppnw:
            mf_loss = tf.reduce_mean((1 + self.ppnw_g * ( self.pi_ui  - self.pi_uj )) * tf.math.softplus(-pos_scores + neg_scores))
        emb_loss = self.decay * regularizer        
        if self.pop_reg:
            reg_loss = self.pop_reg_decay * tf.square(pearson_r(pos_scores, self.pos_i_pop_rev))
        else:
            reg_loss = tf.constant(0.0, tf.float32, [1])
        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items 
    print('n_users', config['n_users'])
    print('n_items', config['n_items'])

    """
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, item_pop_rev, d_adj  = data_generator.get_adj_mat()
    theta_u, theta_i, theta_std_u2, theta_i_p = data_generator.get_ppnw()
    config['theta_u'] = theta_u
    config['theta_i'] = theta_i
    config['theta_std_u2'] = theta_std_u2
    config['theta_i_p'] = theta_i_p


    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    if args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')

    if args.adj_type == 'mean':
        config['norm_adj'] = mean_adj
        print('use the norm_wo_self adjacency matrix')

    config['item_pop_rev'] = item_pop_rev
    config['d'] = d_adj 
    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = AdjNorm(data_config=config, pretrain_data=pretrain_data)

    """
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth = True
    sess = tf.Session(config=config_)

    """
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    degree_loger, cover_loger = [], []
    monitor_list = []
    stopping_step = 0
    should_stop = False
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        
        if args.drop_edge == 1 and (epoch + 1) % 10 ==0:
            sample_adj = data_generator.get_sample_adj_mat(args.drop_edge_percent, args.pop_penalty)
            model.norm_adj = sample_adj
            print('drop edge n_interactions', len(model.norm_adj.nonzero()[0]), model.norm_adj.sum(axis=0) )#len(model.norm_adj.nonzero()[0]))


        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()

            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.node_dropout: eval(args.node_dropout),
                                          model.mess_dropout: eval(args.mess_dropout),
                                          model.neg_items: neg_items})
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch ) % 20 != 0:
            if args.drop_edge ==1:
                start = time()
                sample_adj = data_generator.get_sample_adj_mat(args.drop_edge_percent, args.pop_penalty)
                model.norm_adj= sample_adj

            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                     epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss)
                print(perf_str)
            if args.skip == 1:  
                continue


        t2 = time()

        users_to_val = list(data_generator.val_set.keys())
        degree, cover, ret = test(sess, model, config['norm_adj'], users_to_val, 20, False, False, drop_flag=True)

        if args.monitor == True:
            users_to_test = list(data_generator.test_set.keys())
            degree_m, pru_m, ret_m = test(sess, model, config['norm_adj'], users_to_test, 20, True, True, drop_flag=True)
            monitor_list.append([epoch, ret_m['recall'][1], ret_m['ndcg'][1], degree_m, pru_m])        
        
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'][1])
        pre_loger.append(ret['precision'][1])
        ndcg_loger.append(ret['ndcg'][1])
        hit_loger.append(ret['hit_ratio'][1])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train loss==[%.5f=%.5f + %.5f + %.5f], val recall=[%.5f, %.5f], ' \
                       'val precision=[%.5f, %.5f], val hit=[%.5f, %.5f], val ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, epoch, expected_order='acc', flag_step=m)

        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            model_file = save_saver.save(sess, weights_save_path + '/weights_', global_step=epoch)
            print('save the weights in path: ', weights_save_path)


    save_saver.restore(sess, model_file)
    users_to_test = list(data_generator.test_set.keys())
    degree, cover, ret = test(sess, model, config['norm_adj'], users_to_test, 10, True, True, drop_flag=True)

    print('test recall', ret['recall'], 'test precision', ret['precision'], '\n'
          'test ndcg', ret['ndcg'], 'test hit_ratio', ret['hit_ratio'], 'auc', ret['auc'])


    degree_, cover_, ret_ = test(sess, model, config['norm_adj'], users_to_test, 20, True, True, drop_flag=True)
    degree_, cover_, ret_ = test(sess, model, config['norm_adj'], users_to_test, 50, True, True, drop_flag=True)

    if args.monitor == True:
        with open('/data/GCN_pop_bias/'+str(args.dataset)+'_monitor_recall_degree.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'recall@20', 'ndcg@20', 'novelty@20','pru@20'])
            writer.writerows(monitor_list)
            

   





