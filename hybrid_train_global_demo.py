"""
read in generated paths. Plan path using neural planner.
For failed segments, use informed-rrt* to generate demonstration. When the segment is the
entire path, just use the preloaded data as demonstration.
Load the demonstration into replay memory and reservoid memory
"""
from __future__ import print_function
from Model.GEM_end2end_model import End2EndMPNet
import numpy as np
import argparse
import os
import torch
from data_loader import *
from torch.autograd import Variable
import copy
import os
import random
import time
from utility import *
from plan_general import *
import plan_s2d
def normalize(x, bound):
    # normalize to -1 ~ 1
    return x / bound
DEFAULT_STEP = 0.05
def main(args):
    # set seed
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Create model directory
    md_type = 'deep'
    if not args.AEtype_deep:
        md_type = 'simple'
    # Depending on env type, load the planning function
    if args.env_type == 's2d':
        IsInCollision = plan_s2d.IsInCollision
    elif args.env_type == 'c2d':
        pass
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    mpNet = End2EndMPNet(args.mlp_input_size, args.output_size, md_type, \
                args.n_tasks, args.n_memories, args.memory_strength, args.grad_step)
    model_path='mpNet_cont_train_epoch_%d.pkl' %(args.start_epoch)
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
        mpNet.set_opt(torch.optim.Adagrad, lr=1e-2)
    if args.start_epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))

    # load train and test data
    print('loading...')
    obc,obs,paths,path_lengths = load_raw_dataset(N=args.no_env, NP=args.no_motion_paths, folder=args.data_path)
    obs = torch.from_numpy(obs).type(torch.FloatTensor)
    # Pretrain the Models, we do this only before hybrid training
    # and we don't do this for several epochs

    # set the number of paths trained before hybrid training
    # so that in each epoch, the total number up to now will be shown
    num_path_trained = 0
    num_trained_samples = 0
    data_all = []  # will record all data, even for each epoch
    while num_path_trained < args.pretrain_path:
        # pretrain
        # randomly select one env, and one path
        i = np.random.randint(low=0, high=len(paths))
        j = np.random.randint(low=0, high=len(paths[0]))
        if path_lengths[i][j] == 0:
            # if the length is zero, then no point training on that
            continue
        print('pretraining... env: %d, path: %d' % (i+1, j+1))
        pretrain_path = paths[i][j][:path_lengths[i][j]]  # numpy
        dataset, targets, env_indices = transformToTrain(pretrain_path, \
                                        len(pretrain_path), obs[i], i)
        num_trained_samples += len(targets)
        data_all += list(zip(dataset,targets,env_indices))
        bi = np.concatenate( (obs[i].numpy().reshape(1,-1).repeat(len(dataset),axis=0), dataset), axis=1).astype(np.float32)
        bi = torch.FloatTensor(bi)
        bt = torch.FloatTensor(targets)
        # normalize input and target with world_size
        bi = normalize(bi, args.world_size)
        bt = normalize(bt, args.world_size)
        mpNet.zero_grad()
        bi=to_var(bi)
        bt=to_var(bt)
        mpNet.observe(bi, 0, bt)
        num_path_trained += 1

    print('continual training...')
    for epoch in range(1,args.num_epochs+1):
        # Unlike number of trained paths, we print time for each epoch independently
        time_env = []
        print('epoch' + str(epoch))
        for i in range(len(paths)):
            time_path = []
            for j in range(len(paths[i])):
                time0 = time.time()
                print('epoch: %d, training... env: %d, path: %d' % (epoch, i+1, j+1))
                if path_lengths[i][j] == 0:
                    continue
                fp = False
                path = [torch.from_numpy(paths[i][j][0]).type(torch.FloatTensor),\
                        torch.from_numpy(paths[i][j][path_lengths[i][j]-1]).type(torch.FloatTensor)]
                step_sz = DEFAULT_STEP

                # hybrid train
                # Note that path are list of tensors
                for t in range(args.MAX_NEURAL_REPLAN):
                # adaptive step size on replanning attempts
                    if (t == 2):
                        step_sz = 0.04
                    elif (t == 3):
                        step_sz = 0.03
                    elif (t > 3):
                        step_sz = 0.02
                    unnormalize_func = lambda x: x * args.world_size
                    path = neural_replan(mpNet, path, obc[i], obs[i], IsInCollision, unnormalize_func, step_sz=step_sz)
                    path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                    if feasibility_check(path,obc[i], IsInCollision, step_sz=0.01):
                        fp = True
                        print('feasible, ok!')
                        break
                print('number of samples trained up to now: %d' % (num_trained_samples))
                print('number of paths trained up to now: %d' % (num_path_trained))
                if not fp:
                    print('using demonstration...')
                    # since this is complete replan, we are using the finest step size
                    normalize_func = lambda x: normalize(x, args.world_size)
                    path, added_path = complete_replan_global(mpNet, path, paths[i][j], path_lengths[i][j], \
                                                              obc[i], obs[i], i, normalize_func, step_sz=0.01)
                    data_all += added_path
                    num_trained_samples += len(added_path)
                    num_path_trained += 1
                    # perform rehersal when certain number of batches have passed
                    if num_path_trained % args.freq_rehersal == 0 and len(data_all) > args.batch_rehersal:
                        print('rehersal...')
                        sample = random.sample(data_all, args.batch_rehersal)
                        dataset, targets, env_indices = list(zip(*sample))
                        dataset, targets, env_indices = list(dataset), list(targets), list(env_indices)
                        bi = np.concatenate( (obs[env_indices], dataset), axis=1).astype(np.float32)
                        bt = targets
                        bi = torch.FloatTensor(bi)
                        bt = torch.FloatTensor(bt)
                        bi, bt = normalize(bi, args.world_size), normalize(bt, args.world_size)
                        mpNet.zero_grad()
                        bi=to_var(bi)
                        bt=to_var(bt)
                        mpNet.observe(bi, 0, bt, False)  # train but don't remember
                else:
                    # neural planning is feasible, then we just put that in our data_all list
                    # for rehersal
                    # if include_suc_path is turned on, then add it into all_data for rehersal
                    if args.include_suc_path:
                        # convert path to numpy first for transformation
                        path = [p.numpy() for p in path]
                        dataset, targets, env_indices = transformToTrain(path, \
                                                        len(path), obs[i], i)
                        data_all += list(zip(dataset,targets,env_indices))

                time_spent = time.time() - time0
                time_path.append(time_spent)
                print('it takes time: %f s' % (time_spent))
            time_env.append(time_path)
        print('number of samples trained in total: %d' % (num_trained_samples))

        # Save the models
        if epoch > 0:
            model_path='mpNet_cont_train_epoch_%d.pkl' %(epoch)
            save_state(mpNet, torch_seed, np_seed, py_seed, os.path.join(args.model_path,model_path))
            num_train_sample_record = args.model_path+'num_trained_samples_epoch_%d.txt' % (epoch)
            num_train_path_record = args.model_path+'num_trained_paths_epoch_%d.txt' % (epoch)
            f = open(num_train_sample_record, 'w')
            f.write('%d\n' % (num_trained_samples))
            f.close()
            f = open(num_train_path_record, 'w')
            f.write('%d\n' % (num_path_trained))
            f.close()
            pickle.dump(time_env, open(args.model_path+'planning_time_epoch_%d.txt' % (epoch), "wb" ))
            # test

parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
parser.add_argument('--no_env', type=int, default=100,help='directory for obstacle images')
parser.add_argument('--no_motion_paths', type=int,default=4000,help='number of optimal paths in each environment')
parser.add_argument('--grad_step', type=int, default=1, help='number of gradient steps in continual learning')
# for continual learning
parser.add_argument('--n_tasks', type=int, default=1,help='number of tasks')
parser.add_argument('--n_memories', type=int, default=256, help='number of memories for each task')
parser.add_argument('--memory_strength', type=float, default=0.5, help='memory strength (meaning depends on memory)')
# Model parameters
parser.add_argument('--mlp_input_size', type=int , default=28+4, help='dimension of the input vector')
parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')

parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--seen', type=int, default=0, help='seen or unseen? 0 for seen, 1 for unseen')
parser.add_argument('--AEtype_deep', type=int, default=1, help='indicate that autoencoder is deep model')
parser.add_argument('--device', type=int, default=0, help='cuda device')

parser.add_argument('--batch_path', type=int,default=10,help='number of optimal paths in each environment')
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--freq_rehersal', type=int, default=20, help='after how many paths perform rehersal')
parser.add_argument('--batch_rehersal', type=int, default=100, help='rehersal on how many data (not path)')
parser.add_argument('--data_path', type=str, default='../data/simple/')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--MAX_NEURAL_REPLAN', type=int, default=1)
parser.add_argument('--env_type', type=str, default='s2d')

parser.add_argument('--pretrain_path', type=int, default=200, help='number of paths for pretraining before hybrid train')
parser.add_argument('--include_suc_path', type=int, default=0, help='0 for not including neural path into replay buffer')
parser.add_argument('--world_size', type=int, default=50, help='boundary of world')
args = parser.parse_args()
print(args)
main(args)
