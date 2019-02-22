import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


# DMLP Model-Path Generator
class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 1280),nn.PReLU(),nn.Dropout(),
		nn.Linear(1280, 1024),nn.PReLU(),nn.Dropout(),
		nn.Linear(1024, 768),nn.PReLU(),nn.Dropout(),
		nn.Linear(768, 256),nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 256),nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 128),nn.PReLU(), nn.Dropout(),
		nn.Linear(128, 64),nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32),nn.PReLU(),
		nn.Linear(32, output_size), nn.Tanh())  # added tanh to make sure output is -1 ~ 1

	def forward(self, x):
		out = self.fc(x)
		return out
