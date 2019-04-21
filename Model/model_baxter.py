import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


# # DMLP Model-Path Generator
# class MLP(nn.Module):
# 	def __init__(self, input_size, output_size):
# 		super(MLP, self).__init__()
# 		self.fc = nn.Sequential(
# 			nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
# 			nn.Linear(1280, 896), nn.PReLU(), nn.Dropout(),
# 			nn.Linear(896, 512), nn.PReLU(), nn.Dropout(),
# 			nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
# 			nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
# 			nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
# 			nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
# 			nn.Linear(64, 32), nn.PReLU(),
# 			nn.Linear(32, output_size)
# 			)


# DMLP Model-Path Generator (x2)
# class MLP(nn.Module):
# 	def __init__(self, input_size, output_size):
# 		super(MLP, self).__init__()
# 		self.fc = nn.Sequential(
#                     nn.Linear(input_size, 2560), nn.PReLU(), nn.Dropout(),
#                     nn.Linear(2560, 1792), nn.PReLU(), nn.Dropout(),
#                     nn.Linear(1792, 1024), nn.PReLU(), nn.Dropout(),
#                     nn.Linear(1024, 768), nn.PReLU(), nn.Dropout(),
#                     nn.Linear(768, 512), nn.PReLU(), nn.Dropout(),
#                     nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
#                     nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
#                     nn.Linear(128, 64), nn.PReLU(),
#                     nn.Linear(64, output_size)
#                 )

# 	def forward(self, x):
# 		out = self.fc(x)
# 		out = torch.clamp(out, -1, 1)
# 		return out

# DMLP Model-Path Generator (x3)
class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		self.fc = nn.Sequential(
                    nn.Linear(input_size, 3840), nn.PReLU(), nn.Dropout(),
                    nn.Linear(3840, 2688), nn.PReLU(), nn.Dropout(),
                    nn.Linear(2688, 1536), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1536, 1152), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1152, 768), nn.PReLU(), nn.Dropout(),
                    nn.Linear(768, 384), nn.PReLU(), nn.Dropout(),
                    nn.Linear(384, 192), nn.PReLU(), nn.Dropout(),
                    nn.Linear(192, 96), nn.PReLU(),
                    nn.Linear(96, output_size)
                )

	def forward(self, x):
		out = self.fc(x)
		out = torch.clamp(out, -1, 1)
		return out

# DMLP Model-Path Generator (x4)
class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		self.fc = nn.Sequential(
					nn.Linear(input_size, 5120), nn.PReLU(), nn.Dropout(),
					nn.Linear(5120, 3584), nn.PReLU(), nn.Dropout(),
					nn.Linear(3584, 2048), nn.PReLU(), nn.Dropout(),
					nn.Linear(2048, 1536), nn.PReLU(), nn.Dropout(),
					nn.Linear(1536, 1024), nn.PReLU(), nn.Dropout(),
					nn.Linear(1024, 512), nn.PReLU(), nn.Dropout(),
					nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
					nn.Linear(256, 128), nn.PReLU(),
					nn.Linear(128, output_size)
				)

	def forward(self, x):
		out = self.fc(x)
		out = torch.clamp(out, -1, 1)
		return out


class MLP_Path(nn.Module):
 	def __init__(self, input_size, output_size):
 		super(MLP_Path, self).__init__()
 		self.fc = nn.Sequential(
					nn.Linear(input_size, 2048), nn.PReLU(), nn.Dropout(),
					nn.Linear(2048, 1024), nn.PReLU(), nn.Dropout(),
					nn.Linear(1024, output_size))

 	def forward(self, x):
 		out = self.fc(x)
 		return out
