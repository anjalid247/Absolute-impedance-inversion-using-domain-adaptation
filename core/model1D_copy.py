import torch.nn as nn
from torch.nn.utils import weight_norm
import torch
import numpy as np


#1-D TCN Model from: Mustafa et al.
   
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, self.chomp_size:].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        self.relu = nn.ELU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv1.weight.data)
        nn.init.kaiming_uniform_(self.conv2.weight.data)
        #self.conv1.weight.data.normal_(0, 0.01)
        #self.conv2.weight.data.normal_(0, 0.01)
        #torch.nn.init.xavier_uniform(self.conv1.weight)
        if self.downsample is not None:
            nn.init.kaiming_uniform_(self.downsample.weight.data)
            #self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=7, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=int((kernel_size - 1)/2 * dilation_size), dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.tcn_local = TemporalConvNet(num_inputs=3, num_channels=[3, 6, 6, 6, 6, 6, 5], kernel_size=9, dropout=0.2)
        self.regression = nn.Sequential(nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1))
        

    def forward(self, input):
        #out = self.tcn_local(input[:,:,:,3])  # only uncomment when you are using 2-D dataloaders
        out = self.tcn_local(input)
        out = self.regression(out)
        
        return out
