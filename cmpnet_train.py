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
import Model.model as model
import Model.model_c2d as model_c2d
import Model.model_home as model_home
import Model.AE.CAE_r3d as CAE_r3d
import Model.AE.CAE as CAE_2d
import Model.AE.CAE_simple as CAE_simple
import Model.AE.CAE_home as CAE_home
import Model.AE.CAE_home_voxel_2 as CAE_home_voxel_2
import Model.AE.CAE_home_voxel_3 as CAE_home_voxel_3
import Model.AE.CAE_home_voxel as CAE_home_voxel
import Model.model_c2d_simple as model_c2d_simple
import numpy as np
import argparse
import os
import torch
import data_loader_2d, data_loader_r3d, data_loader_r2d
from torch.autograd import Variable
import copy
import os
import random
from utility import *
import utility_s2d, utility_c2d, utility_r3d, utility_r2d, utility_home


from tensorboardX import SummaryWriter
def main(args):
    # set seed
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    # decide dataloader, MLP, AE based on env_type
    if args.env_type == 's2d':
        load_dataset = data_loader_2d.load_dataset
        normalize = utility_s2d.normalize
        unnormalize = utility_s2d.unnormalize
        CAE = CAE_2d
        MLP = model.MLP
    elif args.env_type == 'c2d':
        load_dataset = data_loader_2d.load_dataset
        normalize = utility_c2d.normalize
        unnormalize = utility_c2d.unnormalize
        CAE = CAE_2d
        MLP = model_c2d_simple.MLP
    elif args.env_type == 'r3d':
        load_dataset = data_loader_r3d.load_dataset
        normalize = utility_r3d.normalize
        unnormalize = utility_r3d.unnormalize
        CAE = CAE_r3d
        MLP = model.MLP
    elif args.env_type == 'r2d':
        load_dataset = data_loader_r2d.load_dataset
        normalize = utility_r2d.normalize
        unnormalize = utility_r2d.unnormalize
        CAE = CAE_2d
        #MLP = model.MLP
        MLP = model_c2d.MLP
        args.world_size = [20., 20., np.pi]
    elif args.env_type == 'r2d_simple':
        load_dataset = data_loader_r2d.load_dataset
        normalize = utility_r2d.normalize
        unnormalize = utility_r2d.unnormalize
        CAE = CAE_2d
        #MLP = model.MLP
        MLP = model_c2d_simple.MLP
        args.world_size = [20., 20., np.pi]
    elif args.env_type == 'home':
        import data_loader_home
        load_dataset = data_loader_home.load_dataset
        normalize = utility_home.normalize
        unnormalize = utility_home.unnormalize
        CAE = CAE_home_voxel_3
        MLP = model_home.MLP
    elif args.env_type == 'home_mlp2':
        import data_loader_home
        load_dataset = data_loader_home.load_dataset
        normalize = utility_home.normalize
        unnormalize = utility_home.unnormalize
        CAE = CAE_home_voxel_3
        MLP = model_home.MLP2

    elif args.env_type == 'home_mlp3':
        import data_loader_home
        load_dataset = data_loader_home.load_dataset
        normalize = utility_home.normalize
        unnormalize = utility_home.unnormalize
        CAE = CAE_home_voxel_3
        MLP = model_home.MLP3

    elif args.env_type == 'home_mlp4':
        import data_loader_home
        load_dataset = data_loader_home.load_dataset
        normalize = utility_home.normalize
        unnormalize = utility_home.unnormalize
        CAE = CAE_home_voxel_2
        MLP = model_home.MLP

    elif args.env_type == 'home_mlp5':
        import data_loader_home
        load_dataset = data_loader_home.load_dataset
        normalize = utility_home.normalize
        unnormalize = utility_home.unnormalize
        CAE = CAE_home_voxel_3
        MLP = model_home.MLP

    if args.memory_type == 'res':
        mpNet = End2EndMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, \
                    args.output_size, 'deep', args.n_tasks, args.n_memories, args.memory_strength, args.grad_step, \
                    CAE, MLP)
    elif args.memory_type == 'rand':
        pass
    # setup loss
    if args.env_type == 's2d':
        loss_f = mpNet.loss
    elif args.env_type == 'c2d':
        loss_f = mpNet.loss
    elif args.env_type == 'r3d':
        loss_f = mpNet.loss
    elif args.env_type == 'r2d':
        loss_f = mpNet.loss
    elif args.env_type == 'r2d_simple':
        loss_f = mpNet.loss
    elif args.env_type == 'home':
        loss_f = mpNet.pose_loss
    elif args.env_type == 'home_mlp2':
        loss_f = mpNet.pose_loss
    elif args.env_type == 'home_mlp3':
        loss_f = mpNet.pose_loss
    elif args.env_type == 'home_mlp4':
        loss_f = mpNet.pose_loss
    elif args.env_type == 'home_mlp5':
        loss_f = mpNet.pose_loss

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
        elif args.opt == 'ASGD':
            mpNet.set_opt(torch.optim.ASGD, lr=args.learning_rate)
    if args.start_epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))

    # load train and test data
    print('loading...')
    obs, path_data = load_dataset(N=args.no_env, NP=args.no_motion_paths, folder=args.data_path)
    # use some path data as validation set
    val_single_path_data = path_data[-100:]
    # for each batch of validation dataset, use 10 paths
    val_path_data = []
    for i in range(len(val_single_path_data)):
        j = 0
        dataset = []
        targets = []
        env_indices = []
        while j < 10:
            idx = (i+j) % len(val_single_path_data)
            p_dataset, p_targets, p_env_indices = val_single_path_data[idx]
            dataset += p_dataset
            targets += p_targets
            env_indices += p_env_indices
            j += 1
        val_path_data.append([dataset, targets, env_indices])
    # set training data again
    path_data = path_data[:-100]

    # print out data length:
    print('length of validation data:')
    print(len(val_path_data))
    print('length of path data:')
    print(len(path_data))


    # Train the Models
    print('training...')
    writer = SummaryWriter('./runs/'+args.env_type)
    record_loss = 0.
    record_i = 0
    val_record_loss = 0.
    val_record_i = 0
    for epoch in range(args.start_epoch+1,args.num_epochs+1):
        data_all = []
        num_path_trained = 0
        print('epoch' + str(epoch))
        dataset, targets, env_indices = [], [], []
        path_ct = 0
        for i in range(0,len(path_data)):
            print('epoch: %d, training... path: %d' % (epoch, i+1))
            p_dataset, p_targets, p_env_indices = path_data[i]
            print(p_dataset)
            if len(p_dataset) == 0:
                # empty path
                continue
            dataset = p_dataset
            targets = p_targets
            env_indices = p_env_indices
            path_ct += 1
            if path_ct % args.train_path != 0:
                continue
            # record
            data_all += list(zip(dataset,targets,env_indices))
            bi = np.array(dataset).astype(np.float32)
            bobs = np.array(obs[env_indices]).astype(np.float32)
            #bi = np.concatenate( (obs[env_indices], dataset), axis=1).astype(np.float32)
            bt = targets
            bi = torch.FloatTensor(bi)
            bobs = torch.FloatTensor(bobs)
            bt = torch.FloatTensor(bt)
            bi, bt = normalize(bi, args.world_size), normalize(bt, args.world_size)
            mpNet.zero_grad()
            bi=to_var(bi)
            bobs = to_var(bobs)
            bt=to_var(bt)
            print('before training losses:')
            print(loss_f(mpNet(bi, bobs), bt))
            mpNet.observe(0, bi, bobs, bt, loss_f=loss_f)
            print('after training losses:')
            loss = loss_f(mpNet(bi, bobs), bt)
            print(loss)

            num_path_trained += 1
            record_loss += loss.data
            record_i += 1
            if record_i % 100 == 0:
                writer.add_scalar('train_loss', record_loss / 100, record_i)
                record_loss = 0.

            ######################################
            # loss on validation set
            # use 10 paths in validation dataset to calculate val loss
            val_dataset, val_targets, val_env_indices = val_path_data[i % len(val_path_data)]
            dataset = val_dataset
            targets = val_targets
            env_indices = val_env_indices
            bi = np.array(dataset).astype(np.float32)
            bobs = np.array(obs[env_indices]).astype(np.float32)
            #bi = np.concatenate( (obs[env_indices], dataset), axis=1).astype(np.float32)
            bt = targets
            bi = torch.FloatTensor(bi)
            bobs = torch.FloatTensor(bobs)
            bt = torch.FloatTensor(bt)
            bi, bt = normalize(bi, args.world_size), normalize(bt, args.world_size)
            bi=to_var(bi)
            bobs = to_var(bobs)
            bt=to_var(bt)
            print('validation losses:')
            loss = loss_f(mpNet(bi, bobs), bt)
            print(loss)

            val_record_loss += loss.data
            val_record_i += 1
            if val_record_i % 100 == 0:
                writer.add_scalar('val_loss', val_record_loss / 100, val_record_i)
                val_record_loss = 0.

            ######################################
            # perform rehersal when certain number of batches have passed
            if args.freq_rehersal and num_path_trained % args.freq_rehersal == 0:
                print('rehersal...')
                sample = random.sample(data_all, args.batch_rehersal)
                dataset, targets, env_indices = list(zip(*sample))
                dataset, targets, env_indices = list(dataset), list(targets), list(env_indices)
                bi = np.array(dataset).astype(np.float32)
                bobs = np.array(obs[env_indices]).astype(np.float32)
                bt = targets
                bi = torch.FloatTensor(bi)
                bobs = torch.FloatTensor(bobs)
                bt = torch.FloatTensor(bt)
                bi, bt = normalize(bi, args.world_size), normalize(bt, args.world_size)
                mpNet.zero_grad()
                bi=to_var(bi)
                bobs = to_var(bobs)
                bt=to_var(bt)
                mpNet.observe(0, bi, bobs, bt, False, loss_f=loss_f)  # train but don't remember
            dataset, targets, env_indices = [], [], []
            path_ct = 0

        # Save the models
        if epoch > 0:
            model_path='cmpnet_epoch_%d.pkl' %(epoch)
            save_state(mpNet, torch_seed, np_seed, py_seed, os.path.join(args.model_path,model_path))
            # test
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

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
parser.add_argument('--total_input_size', type=int, default=2800+4, help='dimension of total input')
parser.add_argument('--AE_input_size', nargs='+', type=int, default=2800, help='dimension of input to AE')
parser.add_argument('--mlp_input_size', type=int , default=28+4, help='dimension of the input vector')
parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')

parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--seen', type=int, default=0, help='seen or unseen? 0 for seen, 1 for unseen')
parser.add_argument('--device', type=int, default=0, help='cuda device')

parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--freq_rehersal', type=int, default=20, help='after how many paths perform rehersal')
parser.add_argument('--batch_rehersal', type=int, default=100, help='rehersal on how many data (not path)')
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
