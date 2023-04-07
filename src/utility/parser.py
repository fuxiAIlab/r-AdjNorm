'''
Tensorflow Implementation of r-Adjnorm model in:
Minghao Zhao  et al. Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering. In SIGIR 2022.

@author: Minghao Zhao(zhaominghao@corp.netease.com)
'''
####################################################
# This section of code adapted from WangXiang/NGCF
###################################################
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run AdjNorm.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='amazon-book',
                        help='Choose a dataset from amazon-book')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64, 64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='AdjNorm',
                        help='Specify the name of model (AdjNorm).')
    parser.add_argument('--adj_type', nargs='?', default='mean',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='lightgcn',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc, lightgcn, lrgccf}.')

    parser.add_argument('--single', type=int, default=0,
                        help='single') 
 
    parser.add_argument('--pop_reg', type=int, default=0,
                        help='whether to enable pop_reg')
    parser.add_argument('--pop_reg_decay', type=float, default=1.,
                        help='the hyperparameter for pop_reg')
    
    parser.add_argument('--drop_edge', type=int, default=0,
                        help='whether to  turn on DropEdge')
    parser.add_argument('--drop_edge_percent', type=float, default=0.5,
                        help='the percent of DropEdge') 
    parser.add_argument('--pop_penalty', type=float, default=0,
                        help='whether to  enable pop_penalty(degree penalty for popoluar nodes) when conducting DropEdge ')     
 
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[10, 20, 30, 40, 50]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    
    parser.add_argument('--skip', type=int, default=0,
                        help='skip the test')
    
    parser.add_argument('--r', type=float, default=0.5, 
                        help='normalization coefficient')
    parser.add_argument('--monitor', type=bool, default=False, 
                        help='monitor the test evaluation')
     
    parser.add_argument('--negative_sample', type=int, default=0)
    parser.add_argument('--positive_sample', type=int, default=0)
    parser.add_argument('--ns', type=float, default=0,
                        help='the hyperparameter for  negative sampling')
   
    parser.add_argument('--ppnw', type=int, default=0, 
                        help='the hyperparameter for  ppnw') 
    parser.add_argument('--ppnw_a', type=float, default=1, 
                        help='the hyperparameter for  ppnw')
    parser.add_argument('--ppnw_g', type=float, default=1, 
                        help='the hyperparameter for  ppnw')
    parser.add_argument('--ppnw_l', type=float, default=1, 
                        help='the hyperparameter for  ppnw')
  
    parser.add_argument('--pc', type=int, default=0, 
                        help='the hyperparameter for  pc')
    parser.add_argument('--pc_a', type=float, default=1, 
                        help='the hyperparameter for  pc')
    parser.add_argument('--pc_b', type=float, default=0.5, 
                        help='the hyperparameter for  pc')
    return parser.parse_args()
