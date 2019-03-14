import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 1280),nn.PReLU(),nn.Dropout(),
		nn.Linear(1280, 1024),nn.PReLU(),nn.Dropout(),
		nn.Linear(1024, 896),nn.PReLU(),nn.Dropout(),
		nn.Linear(896, 768),nn.PReLU(),nn.Dropout(),
		nn.Linear(768, 512),nn.PReLU(),nn.Dropout(),
		nn.Linear(512, 384),nn.PReLU(),nn.Dropout(),
		nn.Linear(384, 256),nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 256),nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 128),nn.PReLU(), nn.Dropout(),
		nn.Linear(128, 64),nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32),nn.PReLU(),
		nn.Linear(32, output_size))
	def init_weights(m):
	    if type(m) == nn.Linear:
	        torch.nn.init.xavier_uniform(m.weight)
	        m.bias.data.fill_(0.01)
		self.fc.apply(init_weights)
	def forward(self, x):
		out = self.fc(x)
		return out
