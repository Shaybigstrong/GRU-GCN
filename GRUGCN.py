#Import related libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch as th
import torch.nn as nn
import dgl
from dgl import function as fn
import pandas as pd
import numpy as np
import networkx as nx
import torch.nn.functional as F
from torch.autograd import Variable
import time


class TAGConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=2,
                 bias=True,
                 activation=None):

        super(TAGConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k
        self._activation = activation

        self.lin = nn.Linear(in_feats * (self._k + 1), out_feats, bias=bias)
        # self.dropout = nn.Dropout(0.5)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)

    def forward(self, graph, feat):
        graph = graph.local_var()

        norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)

        shp = norm.shape + (1,) * (feat.dim() - 1)

        norm = th.reshape(norm, shp).to(feat.device)

        # D-1/2 A D -1/2 X
        fstack = [feat]
        for _ in range(self._k):
            rst = fstack[-1] * norm

            graph.ndata['h'] = rst

            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))

            rst = graph.ndata['h']
            rst = rst * norm
            fstack.append(rst)

        rst = self.lin(th.cat(fstack, dim=-1))

        if self._activation is not None:
            rst = self._activation(rst)

        return rst


class GCN(nn.Module):

    def __init__(self, input_dim, hidden_size, num_classes):
        super(GCN, self).__init__()

        self.GRU = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=1)

        self.gcn1 = TAGConv(12, hidden_size, k=1)

        self.gcn2 = TAGConv(hidden_size, num_classes, k=1)

    def forward(self, graph, feature):
        h = self.GRU(feature)

        h = F.relu(self.gcn1(graph, h[
            0]))  # Hidden layer operation, absorbing information from neighboring nodes in the first layer

        h = self.gcn2(graph,
                      h)  # Fully connected layer operation, absorbing information from neighboring nodes in the second layer
        return h
