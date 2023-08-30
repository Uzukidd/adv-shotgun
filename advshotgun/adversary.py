import torch
import torch.nn as nn
import torch.nn.functional as F

class adversary_base:
 
    def __init__(self, x:torch.Tensor, batch_size:int=1, victim_model:nn.Module=None):
        self.batch_size = batch_size
        self.original_x = x.detach().clone()
        self.adversarial_x = x.detach().clone()
        self.victim_model = None

class adversary_image(adversary_base):
    
    def __init__(self, x:torch.Tensor, batch_size:int=1):
        super().__init__(x, batch_size)
        

class adversary_pcs(adversary_base):
    
    def __init__(self, x:torch.Tensor, batch_size:int=1):
        super().__init__(x, batch_size)