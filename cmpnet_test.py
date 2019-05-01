'''
This is the main file to run gem_end2end network.
It simulates the real scenario of observing a data, puts it inside the memory (or not),
and trains the network using the data
after training at each step, it will output the R matrix described in the paper
https://arxiv.org/abs/1706.08840
and after sevral training steps, it needs to store the parameter in case emergency
happens
To make it work in a real-world scenario, it needs to listen to the observer at anytime,
and call the network to train if a new data is available
(this thus needs to use multi-process)
here for simplicity, we just use single-process to simulate this scenario
'''
from __future__ import print_function
from Model.GEM_end2end_model import End2EndMPNet
#from GEM_end2end_model_rand import End2EndMPNet as End2EndMPNet_rand
import Model.model as model
import Model.model_c2d as model_c2d
import Model.AE.CAE_r3d as CAE_r3d
import Model.AE.CAE as CAE_2d
#import Model.AE.CAE_complex as CAE_2d
import Model.AE.CAE_simple as CAE_simple
import Model.model_c2d_simple as model_c2d_simple
import numpy as np
import argparse
import os
import torch
from gem_eval import eval_tasks
import plan_s2d, plan_c2d, plan_r3d, plan_r2d
import data_loader_2d, data_loader_r3d, data_loader_r2d
from torch.autograd import Variable
import copy
import os
import gc
import random
from utility import *
import utility_s2d, utility_c2d, utility_r3d, utility_r2d
def main(args):
    # set seed
    print(args.model_path)
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # setup evaluation function and load function
    if args.env_type == 's2d':
        IsInCollision = plan_s2d.IsInCollision
        load_test_dataset = data_loader_2d.load_test_dataset
        normalize = utility_s2d.normalize
        unnormalize = utility_s2d.unnormalize
        CAE = CAE_2d
        MLP = model.MLP
    elif args.env_type == 'c2d':
        IsInCollision = plan_c2d.IsInCollision
        load_test_dataset = data_loader_2d.load_test_dataset
        normalize = utility_c2d.normalize
        unnormalize = utility_c2d.unnormalize
        CAE = CAE_2d
        MLP = model_c2d.MLP
    elif args.env_type == 'r3d':
        IsInCollision = plan_r3d.IsInCollision
        load_test_dataset = data_loader_r3d.load_test_dataset
        normalize = utility_r3d.normalize
        unnormalize = utility_r3d.unnormalize
        CAE = CAE_r3d
        MLP = model.MLP
    elif args.env_type == 'r2d':
        IsInCollision = plan_r2d.IsInCollision
        load_test_dataset = data_loader_r2d.load_test_dataset
        normalize = utility_r2d.normalize
        unnormalize = utility_r2d.unnormalize
        CAE = CAE_2d
        #MLP = model.MLP
        MLP = model_c2d_simple.MLP
        args.world_size = [20., 20., np.pi]
    elif args.env_type == 'r2d_simple':
        IsInCollision = plan_r2d.IsInCollision
        load_test_dataset = data_loader_r2d.load_test_dataset
        normalize = utility_r2d.normalize
        unnormalize = utility_r2d.unnormalize
        CAE = CAE_2d
        #MLP = model.MLP
        MLP = model_c2d_simple.MLP
        args.world_size = [20., 20., np.pi]

    if args.memory_type == 'res':
        mpNet = End2EndMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, \
                    args.output_size, 'deep', args.n_tasks, args.n_memories, args.memory_strength, args.grad_step, \
                    CAE, MLP)
    elif args.memory_type == 'rand':
        #mpNet = End2EndMPNet_rand(args.mlp_input_size, args.output_size, 'deep', \
        #            args.n_tasks, args.n_memories, args.memory_strength, args.grad_step)
        pass
    # load previously trained model if start epoch > 0
    model_path='cmpnet_epoch_%d.pkl' %(args.start_epoch)
    if args.start_epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)
    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
        if args.opt == 'Adagrad':
            mpNet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
        elif args.opt == 'Adam':
            mpNet.set_opt(torch.optim.Adam, lr=args.learning_rate)
        elif args.opt == 'SGD':
            mpNet.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
    if args.start_epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))


    # load train and test data
    print('loading...')
    seen_test_data = load_test_dataset(N=args.seen_N, NP=args.seen_NP, s=args.seen_s, sp=args.seen_sp, folder=args.data_path)
    unseen_test_data = load_test_dataset(N=args.unseen_N, NP=args.unseen_NP, s=args.unseen_s, sp=args.unseen_sp, folder=args.data_path)
    # test
    # testing
    print('testing...')
    seen_test_suc_rate = 0.
    unseen_test_suc_rate = 0.
    T = 1
    for _ in range(T):
        # unnormalize function
        normalize_func=lambda x: normalize(x, args.world_size)
        unnormalize_func=lambda x: unnormalize(x, args.world_size)
        # seen
        time_file = os.path.join(args.model_path,'time_seen_epoch_%d_mlp.p' % (args.start_epoch))
        fes_path_, valid_path_ = eval_tasks(mpNet, seen_test_data, time_file, IsInCollision, normalize_func, unnormalize_func, True)
        valid_path = valid_path_.flatten()
        fes_path = fes_path_.flatten()   # notice different environments are involved
        seen_test_suc_rate += fes_path.sum() / valid_path.sum()
        # unseen
        time_file = os.path.join(args.model_path,'time_unseen_epoch_%d_mlp.p' % (args.start_epoch))
        fes_path_, valid_path_ = eval_tasks(mpNet, unseen_test_data, time_file, IsInCollision, normalize_func, unnormalize_func, True)
        valid_path = valid_path_.flatten()
        fes_path = fes_path_.flatten()   # notice different environments are involved
        unseen_test_suc_rate += fes_path.sum() / valid_path.sum()
    seen_test_suc_rate = seen_test_suc_rate / T
    unseen_test_suc_rate = unseen_test_suc_rate / T    # Save the models
    f = open(os.path.join(args.model_path,'seen_accuracy_epoch_%d.txt' % (args.start_epoch)), 'w')
    f.write(str(seen_test_suc_rate))
    f.close()
    f = open(os.path.join(args.model_path,'unseen_accuracy_epoch_%d.txt' % (args.start_epoch)), 'w')
    f.write(str(unseen_test_suc_rate))
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
    parser.add_argument('--seen_N', type=int, default=0)
    parser.add_argument('--seen_NP', type=int, default=0)
    parser.add_argument('--seen_s', type=int, default=0)
    parser.add_argument('--seen_sp', type=int, default=0)
    parser.add_argument('--unseen_N', type=int, default=0)
    parser.add_argument('--unseen_NP', type=int, default=0)
    parser.add_argument('--unseen_s', type=int, default=0)
    parser.add_argument('--unseen_sp', type=int, default=0)
    parser.add_argument('--grad_step', type=int, default=1, help='number of gradient steps in continual learning')
    # for continual learning
    parser.add_argument('--n_tasks', type=int, default=1,help='number of tasks')
    parser.add_argument('--n_memories', type=int, default=256, help='number of memories for each task')
    parser.add_argument('--memory_strength', type=float, default=0.5, help='memory strength (meaning depends on memory)')
    # Model parameters
    parser.add_argument('--total_input_size', type=int, default=2800+4, help='dimension of total input')
    parser.add_argument('--AE_input_size', type=int, default=2800, help='dimension of input to AE')
    parser.add_argument('--mlp_input_size', type=int , default=28+4, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--data_path', type=str, default='../data/simple/')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--memory_type', type=str, default='res', help='res for reservoid, rand for random sampling')
    parser.add_argument('--env_type', type=str, default='s2d', help='s2d for simple 2d, c2d for complex 2d')
    parser.add_argument('--world_size', nargs='+', type=float, default=20., help='boundary of world')
    parser.add_argument('--opt', type=str, default='Adagrad')
    parser.add_argument('--train_path', type=int, default=1)
    args = parser.parse_args()
    print(args)
    main(args)
