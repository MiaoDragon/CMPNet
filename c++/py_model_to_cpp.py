from __future__ import division
"""
convert from trained python module to C++ pytorch module.
Since we can't use Pickle in C++ to load trained model.
Adapted from work from Anthony Simeonov.
"""
import sys
sys.path.insert(0, "../")
import argparse
from utility import *
from Model.GEM_end2end_model import End2EndMPNet
import Model.model_home as model_home
import Model.AE.CAE_home_voxel_3 as CAE_home_voxel_3
import data_loader_home
from utility import *
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
class Encoder_home(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    def __init__(self, input_size=32, output_size=64):
        super(Encoder, self).__init__()
        input_size = [input_size, input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
            nn.PReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
            nn.PReLU()
        )
        x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            #nn.Linear(first_fc_in_features, 128),
            #nn.PReLU(),
            nn.Linear(first_fc_in_features, output_size)
        )
    @torch.jit.script_method
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class Encoder_home_Annotated(torch.jit.ScriptModule):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    __constants__ = ['encoder', 'head', 'device']
    def __init__(self, input_size=32, output_size=64):
        super(Encoder_home_Annotated, self).__init__()
        input_size = [input_size, input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=6, stride=2),
            nn.PReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
            nn.PReLU()
        )
        x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, output_size)
        )
        self.device = torch.device('cuda')
    @torch.jit.script_method
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class MLP_home(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_home, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, 2560), nn.PReLU())
        self.fc2 = nn.Sequential(nn.Linear(2560, 1024), nn.PReLU())
        self.fc3 = nn.Sequential(nn.Linear(1024, 512), nn.PReLU())
        self.fc4 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
        self.fc5 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
        self.fc6 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
        self.fc7 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return x


class MLP_home_Annotated(torch.jit.ScriptModule):
    __constants__ = ['fc1','fc2','fc3','fc4','fc5','fc6','device']
    def __init__(self, input_size, output_size):
        super(MLP_home_Annotated, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, 2560), nn.PReLU())
        self.fc2 = nn.Sequential(nn.Linear(2560, 1024), nn.PReLU())
        self.fc3 = nn.Sequential(nn.Linear(1024, 512), nn.PReLU())
        self.fc4 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
        self.fc5 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
        self.fc6 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
        self.fc7 = nn.Linear(64, output_size)

        self.device = torch.device('cuda')
    @torch.jit.script_method
    def forward(self, x):
        prob = 0.5

        p = 1 - prob
        scale = 1.0/p
        drop1 = (scale)*torch.bernoulli(torch.full((1, 2560), p)).to(device=self.device)
        drop2 = (scale)*torch.bernoulli(torch.full((1, 1024), p)).to(device=self.device)
        drop3 = (scale)*torch.bernoulli(torch.full((1, 512), p)).to(device=self.device)
        drop4 = (scale)*torch.bernoulli(torch.full((1, 256), p)).to(device=self.device)
        drop5 = (scale)*torch.bernoulli(torch.full((1, 128), p)).to(device=self.device)

        out1 = self.fc1(x)
        out1 = torch.mul(out1, drop1)

        out2 = self.fc2(out1)
        out2 = torch.mul(out2, drop2)

        out3 = self.fc3(out2)
        out3 = torch.mul(out3, drop3)

        out4 = self.fc4(out3)
        out4 = torch.mul(out4, drop4)

        out5 = self.fc5(out4)
        out5 = torch.mul(out5, drop5)

        out6 = self.fc6(out5)

        out7 = self.fc7(out6)

        return out7

def copyMLP(MLP_to_copy, mlp_weights):
    # this function is where weights are manually copied from the originally trained
    # MPNet models (which have different naming convention for the weights that doesn't
    # work with manual dropout implementation) into the models defined in this script
    # which have the new layer naming convention

    # mlp_weights is just a state_dict() with the good model weights, not loaded into a particular model yet
    # MLP_to_copy is one of the MLP_Python models defined above (depending on 1.0 or 2.0)
    print(MLP_to_copy.state_dict().keys())
    MLP_to_copy.state_dict()['fc1.0.weight'].copy_(mlp_weights['fc.0.weight'])
    MLP_to_copy.state_dict()['fc2.0.weight'].copy_(mlp_weights['fc.3.weight'])
    MLP_to_copy.state_dict()['fc3.0.weight'].copy_(mlp_weights['fc.6.weight'])
    MLP_to_copy.state_dict()['fc4.0.weight'].copy_(mlp_weights['fc.9.weight'])
    MLP_to_copy.state_dict()['fc5.0.weight'].copy_(mlp_weights['fc.12.weight'])
    MLP_to_copy.state_dict()['fc6.0.weight'].copy_(mlp_weights['fc.15.weight'])
    MLP_to_copy.state_dict()['fc7.weight'].copy_(mlp_weights['fc.17.weight'])

    MLP_to_copy.state_dict()['fc1.0.bias'].copy_(mlp_weights['fc.0.bias'])
    MLP_to_copy.state_dict()['fc2.0.bias'].copy_(mlp_weights['fc.3.bias'])
    MLP_to_copy.state_dict()['fc3.0.bias'].copy_(mlp_weights['fc.6.bias'])
    MLP_to_copy.state_dict()['fc4.0.bias'].copy_(mlp_weights['fc.9.bias'])
    MLP_to_copy.state_dict()['fc5.0.bias'].copy_(mlp_weights['fc.12.bias'])
    MLP_to_copy.state_dict()['fc6.0.bias'].copy_(mlp_weights['fc.15.bias'])
    MLP_to_copy.state_dict()['fc7.bias'].copy_(mlp_weights['fc.17.bias'])

    # PReLU
    MLP_to_copy.state_dict()['fc1.1.weight'].copy_(mlp_weights['fc.1.weight'])
    MLP_to_copy.state_dict()['fc2.1.weight'].copy_(mlp_weights['fc.4.weight'])
    MLP_to_copy.state_dict()['fc3.1.weight'].copy_(mlp_weights['fc.7.weight'])
    MLP_to_copy.state_dict()['fc4.1.weight'].copy_(mlp_weights['fc.10.weight'])
    MLP_to_copy.state_dict()['fc5.1.weight'].copy_(mlp_weights['fc.13.weight'])
    MLP_to_copy.state_dict()['fc6.1.weight'].copy_(mlp_weights['fc.16.weight'])
    return MLP_to_copy

def main(args):
    # Set this value to export models for continual learning or batch training

    load_dataset = data_loader_home.load_dataset
    # Get the right architecture which was used for continual learning
    CAE = CAE_home_voxel_3
    mlp = model_home.MLP2

    # make the big model
    mpNet = End2EndMPNet(total_input_size=150008, AE_input_size=[1,32,32,32], mlp_input_size=78, \
                output_size=7, AEtype='deep', n_tasks=1, n_memories=1, memory_strength=0.5, grad_step=1, \
                CAE=CAE, MLP=mlp)
    # The model that performed well originally, load into the big end2end model
    model_path = args.model_path+args.model_name
    load_net_state(mpNet, model_path)

    # Get the weights from this model and create a copy of the weights in mlp_weights (to be copied over)
    MLP2 = mpNet.mlp
    MLP2.cuda()
    mlp_weights = MLP2.state_dict()

    # Save a copy of the encoder's state_dict() for loading into the annotated encoder later on
    encoder_to_copy = mpNet.encoder
    #encoder_to_copy.cuda()
    torch.save(encoder_to_copy.state_dict(), 'encoder2_save.pkl')

    # do everything for the MLP on the GPU
    device = torch.device('cuda:%d'%(args.device))

    encoder = Encoder_home_Annotated()
    #encoder.cuda()
    # Create the annotated model
    MLP = MLP_home_Annotated(78,7)
    MLP.cuda()

    # Create the python model with the new layer names
    MLP_to_copy = MLP_home(78,7)
    MLP_to_copy.cuda()

    # Copy over the mlp_weights into the Python model with the new layer names
    MLP_to_copy = copyMLP(MLP_to_copy, mlp_weights)

    print("Saving models...")

    # Load the encoder weights onto the gpu and then save the Annotated model
    encoder.load_state_dict(torch.load('encoder2_save.pkl', map_location='cpu'))
    encoder.save("encoder_annotated_test_cpu_2.pt")

    # Save the Python model with the weights copied over and the new layer names in a temp file
    torch.save(MLP_to_copy.state_dict(), 'mlp_no_dropout.pkl')

    # Because the layer names now match, can immediately load this state_dict() into the annotated model and then save it
    MLP.load_state_dict(torch.load('mlp_no_dropout.pkl', map_location=device))
    MLP.save("mlp_annotated_test_gpu_2.pt")

    # Everything from here below just tests both models to see if the outputs match
    obs, path_data = load_dataset(N=1, NP=1, folder=args.data_path)


    # write test case
    obs_test = np.array([0.1,1.2,3.0,2.5,1.4,5.2,3.4,-1.])
    #obs_test = obs_test.reshape((1,2,2,2))
    np.savetxt('obs_voxel_test.txt', obs_test, delimiter='\n', fmt='%f')

    # write obstacle to flattened vector representation, then later be loaded in the C++
    obs_out = obs.flatten()
    np.savetxt('obs_voxel.txt', obs_out, delimiter='\n', fmt='%f')


    obs = torch.from_numpy(obs)
    obs = Variable(obs)
    # h = mpNet.encoder(obs)
    h = encoder(obs)
    path_data = np.array([-0.08007369,  0.32780212, -0.01338363,  0.00726194, 0.00430644, -0.00323558,
                       0.18593094,  0.13094018, 0.18499476, 0.3250918, 0.52175426, 0.07388325, -0.49999127, 0.52322733])

    path_data = torch.from_numpy(path_data).type(torch.FloatTensor)

    test_input = torch.cat((path_data, h.data.cpu())).cuda()  # for MPNet1.0
    test_input = Variable(test_input)
    for i in range(5):
        test_output = mpNet.mlp(test_input)
        test_output_save = MLP(test_input)
        print("output %d: " % i)
        print(test_output.data)
        print(test_output_save.data)

parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_path', type=str, default='/media/arclabdl1/HD1/YLmiao/results/CMPnet_res/home_mlp2_lr025_SGD/',help='path for saving trained models')
parser.add_argument('--device', type=int, default=0, help='cuda device')
parser.add_argument('--data_path', type=str, default='/media/arclabdl1/HD1/YLmiao/data/home/')
parser.add_argument('--model_name' ,type=str, default='cmpnet_epoch_100.pkl')
args = parser.parse_args()
print(args)
main(args)
