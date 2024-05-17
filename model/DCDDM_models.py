import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from torch.autograd import Variable
import sys
import yaml
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch_geometric.nn import GATConv
import copy
import time
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from utils import *
from fast_pytorch_kmeans import KMeans
from .TSmodels import get_network

class Dualmodel(nn.Module):
    def __init__(self, args):
        super(Dualmodel, self).__init__()
        self.args = args
        
        self.t_model = get_network(args).to(args.device)
        self.t_model.train()

        self.f_model = get_network(args).to(args.device)
        self.f_model.train()
        
        self.final_rep = self.t_model.final_rep
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.final_rep * 2, self.final_rep),
        #     nn.ReLU(),
        #     nn.Linear(self.final_rep, args.num_classes)
        # ).to(args.device)
        self.mlp = nn.Linear(self.final_rep * 2, args.num_classes).to(args.device)

        
    def forward(self, x):
        # x : [B, C, L]
        x_f = torch.fft.rfft(x, dim=-1)
        x_f = torch.view_as_real(x_f).reshape(x_f.shape[0], x_f.shape[1], -1)
        x_f = x_f[:,:, :x.shape[-1]]

        t_emb = self.t_model.embed(x)
        f_emb = self.f_model.embed(x_f)
        
        emb = torch.cat([t_emb, f_emb], dim=-1)
        
        out = self.mlp(emb)
        # print("x shape : {}, x_f shape : {}, t_emb shape : {}, f_emb shape : {}".format(x.shape, x_f.shape, t_emb.shape, f_emb.shape))

        return out, t_emb, f_emb