from advshotgun import adversary_image
from advshotgun import adversary_pcs

import torch
import torch.nn as nn
import torch.nn.functional as F

class FGSM(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, pert, model):
        pass
    
    
    