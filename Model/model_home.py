import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 2560), nn.PReLU(), nn.Dropout(),
                    nn.Linear(2560, 1792), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1792, 1024), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1024, 768), nn.PReLU(), nn.Dropout(),
                    nn.Linear(768, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                    nn.Linear(128, 64), nn.PReLU(),
                    nn.Linear(64, output_size)
                )

    def forward(self, x):
        out = self.fc(x)
        return out
