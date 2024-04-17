'''
Created on Dec 9, 2015

@author: donghyun
'''
import argparse
import sys
from data_manager import Data_Factory



parser = argparse.ArgumentParser()

# Option for pre-processing data and running ConvMF
parser.add_argument("-d", "--data_path", type=str,
                    help="Path to training, valid and test data sets")
parser.add_argument("-a", "--aux_path", type=str, help="Path to R, D_all sets")

# Option for running ConvMF
parser.add_argument("-o", "--res_dir", type=str,
                    help="Path to ConvMF's result")
parser.add_argument("-e", "--emb_dim", type=int,
                    help="Size of latent dimension for word vectors (default: 200)", default=200)
parser.add_argument("-p", "--pretrain_w2v", type=str,
                    help="Path to pretrain word embedding model  to initialize word vectors")
parser.add_argument("-g", "--give_item_weight", type=bool,
                    help="True or False to give item weight of ConvMF (default = False)", default=True)
parser.add_argument("-k", "--dimension", type=int,
                    help="Size of latent dimension for users and items (default: 50)", default=50)
parser.add_argument("-u", "--lambda_u", type=float,
                    help="Value of user regularizer")
parser.add_argument("-v", "--lambda_v", type=float,
                    help="Value of item regularizer")
parser.add_argument("-n", "--max_iter", type=int,
                    help="Value of max iteration (default: 200)", default=200)
parser.add_argument("-w", "--num_kernel_per_ws", type=int,
                    help="Number of kernels per window size for CNN module (default: 100)", default=100)

args = parser.parse_args()
do_preprocess = True
data_path = 'data/data'
aux_path = 'data/aux'

data_factory = Data_Factory()

res_dir = 'result'
emb_dim = 300
pretrain_w2v = 'glove/glove.6B.300d.txt'
dimension = 50
lambda_u = 0.5
lambda_v = 0.5
max_iter = 300
num_kernel_per_ws = 100
give_item_weight = True

print ("===================================ConvMF Option Setting===================================")
print ("\taux path - %s" % aux_path)
print ("\tdata path - %s" % data_path)
print ("\tresult path - %s" % res_dir)
print ("\tpretrained w2v data path - %s" % pretrain_w2v)
print ("\tdimension: %d\n\tlambda_u: %.4f\n\tlambda_v: %.4f\n\tmax_iter: %d\n\tnum_kernel_per_ws: %d" % (dimension, lambda_u, lambda_v, max_iter, num_kernel_per_ws))
print ("===========================================================================================")

R, D_all = data_factory.load(aux_path)
CNN_X = D_all['X_sequence']
vocab_size = len(D_all['X_vocab']) + 1

from models import ConvMF

if pretrain_w2v is None:
    init_W = None
else:
    init_W = data_factory.read_pretrained_word2vec(pretrain_w2v, D_all['X_vocab'], emb_dim)

train_user = data_factory.read_rating(data_path + '/train_user.dat')
train_item = data_factory.read_rating(data_path + '/train_item.dat')
valid_user = data_factory.read_rating(data_path + '/valid_user.dat')
test_user = data_factory.read_rating(data_path + '/test_user.dat')

ConvMF(max_iter=max_iter, res_dir=res_dir,lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension, vocab_size=vocab_size, init_W=init_W, give_item_weight=give_item_weight, CNN_X=CNN_X, emb_dim=emb_dim, num_kernel_per_ws=num_kernel_per_ws,train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R)
