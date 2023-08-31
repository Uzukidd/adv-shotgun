import torch
import torch.nn as nn
import torch.nn.functional as F

from advshotgun.utils import norm_lp_loss

class adversary_base:
 
    def __init__(self, x:torch.Tensor, ground_truth:torch.Tensor, batch_size:int=1):
        self.batch_size = batch_size
        self.original_x = x.clone() # [b, D...]
        self.adversarial_x = x.clone() # [b, D...]
        self.ground_truth = ground_truth.clone() # [b, D...]
        self.victim_model = None
        self.attack_method = None

class adversary_image(adversary_base):
    
    def __init__(self, x:torch.Tensor, ground_truth:torch.Tensor, batch_size:int=1):
        super().__init__(x, ground_truth, batch_size)
    
    def set_attack_method(self, attack_method:nn.Module):
        self.attack_method = attack_method
        
    def set_victim_model(self, victim_model:nn.Module):
        self.victim_model = victim_model
        
    def evaluate(self):
        assert self.attack_method is not None
        assert self.victim_model is not None
        
        gt_res = self.ground_truth.argmax(1)
        
        pred_logits = self.victim_model(self.original_x)
        pred_res = pred_logits.argmax(1)
        
        adv_x = self.attack_method(self.original_x, self.adversarial_x, self.ground_truth, self.victim_model)
        pred_logits = self.victim_model(adv_x)
        adv_res = pred_logits.argmax(1)
        
        print(f"accuracy:{torch.eq(gt_res, pred_res).float().mean()}")
        print("->")
        print(f"accuracy:{torch.eq(adv_res, pred_res).float().mean()}")
        
        print(f"L2 norm:{norm_lp_loss(self.original_x, adv_x).mean()}")

class adversary_pcs(adversary_base):
    
    def __init__(self, x:torch.Tensor, ground_truth:torch.Tensor, batch_size:int=1):
        super().__init__(x, ground_truth, batch_size)
    
    def set_attack_method(self, attack_method:nn.Module):
        self.attack_method = attack_method
        
    def set_victim_model(self, victim_model:nn.Module):
        self.victim_model = victim_model
        
    def evaluate(self):
        assert self.attack_method is not None
        assert self.victim_model is not None
        
        gt_res = self.ground_truth.argmax(1)
        
        pred_logits = self.victim_model(self.original_x)
        pred_res = pred_logits.argmax(1)
        
        adv_x = self.attack_method(self.original_x, self.adversarial_x, self.ground_truth, self.victim_model)
        pred_logits = self.victim_model(adv_x)
        adv_res = pred_logits.argmax(1)
        
        print(f"accuracy:{torch.eq(gt_res, pred_res).float().mean()}")
        print("->")
        print(f"accuracy:{torch.eq(adv_res, pred_res).float().mean()}")
        
        print(f"L2 norm:{norm_lp_loss(self.original_x, adv_x).mean()}")