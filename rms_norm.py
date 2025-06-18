
from torch import nn
import torch


class RMSNorm(nn.Module):
    def __init__(self,eps:float):
        super().__init__()
        self.eps=eps
    def forward(self,x:torch.Tensor):
        rms=torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
        return x/rms

