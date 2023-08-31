from advshotgun import adversary_image
from advshotgun import adversary_pcs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class randomize_attack(nn.Module):
    
    def __init__(self, eps:float = 0.03):
        super().__init__()
        self.eps = eps
        
    def quantify(self, x):
        return torch.clamp(x, 0.0, 1.0)
        
    def forward(self, x:torch.Tensor, adv_x:torch.Tensor, gt:torch.Tensor, model:nn.Module):
        randomized_grad = torch.randn_like(adv_x)
        
        adv_x = adv_x + (self.eps/1.0) * torch.sign(randomized_grad)
        adv_x = self.quantify(adv_x)
        
        return adv_x
        