import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math 
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import  ChebConv


class ChenNet(MessagePassing):
    
    def __init__(self, input_number, output_number, K=5):
        super(ChenNet, self).__init__()
        # class ChebConv(in_channels: int, out_channels: int, K: int, normalization: Optional[str] = 'sym', bias: bool = True, **kwargs)
        self.conv1 = ChebConv(input_number, 32, K=K, normalization='sym')
        self.conv2 = ChebConv(32, output_number, K=K, normalization='sym')

    def forward(self, node_data, edge_index, edge_weight, lambda_max):
        # node features :math:`(|\mathcal{V}|, F_{in})`,
        # edge indices :math:`(2, |\mathcal{E}|)`,s
        # edge weights :math:`(|\mathcal{E}|)` *(optional)*
        x1 = self.conv1(x=node_data, edge_index=edge_index, edge_weight=edge_weight, lambda_max= lambda_max)
        x1 = F.relu(x1)
        x2 = self.conv2(x=x1, edge_index=edge_index, edge_weight=edge_weight,  lambda_max= lambda_max)
        x2 = nn.LogSoftmax(dim=1)(x2)
        return x2


    