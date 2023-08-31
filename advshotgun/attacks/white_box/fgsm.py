from advshotgun import adversary_image
from advshotgun import adversary_pcs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FGSM(nn.Module):
    
    def __init__(self, eps:float = 0.03):
        super().__init__()
        self.eps = eps
        self.criterion = nn.CrossEntropyLoss()
        
    def quantify(self, x):
        return torch.clamp(x, 0.0, 1.0)
        
    def forward(self, x:torch.Tensor, adv_x:torch.Tensor, gt:torch.Tensor, model:nn.Module):
        adv_x = adv_x.detach()
        adv_x.requires_grad_(True)
        
        logits = model(adv_x)
        loss = self.criterion(logits, gt)
        loss.backward()
        
        adv_x = adv_x + (self.eps/1.0) * torch.sign(adv_x.grad)
        adv_x = self.quantify(adv_x)
        
        return adv_x
        
        
    
    
    
    
    