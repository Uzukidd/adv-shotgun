from advshotgun import adversary_image
from advshotgun import adversary_pcs

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SIADV(nn.Module):
    
    def __init__(self):
        """
        White-box shape-invariant attack.

        Parameters
        ----------

        Returns
        -------
        adv_x : torch.Tensor
            Adversarial sample of x


        >>> SIADV()
        """
        super().__init__()
        
    def quantify(self, x):
        return torch.clamp(x, 0.0, 1.0)
        
    def get_spin_axis_matrix(self, normal_vec):
        """Calculate the spin-axis matrix.

        Args:
            normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
        """
        _, N, _ = normal_vec.shape
        x = normal_vec[:,:,0] # [1, N]
        y = normal_vec[:,:,1] # [1, N]
        z = normal_vec[:,:,2] # [1, N]
        assert abs(normal_vec).max() <= 1
        u = torch.zeros(1, N, 3, 3).cuda()
        denominator = torch.sqrt(1-z**2) # \sqrt{1-z^2}, [1, N]
        u[:,:,0,0] = y / denominator
        u[:,:,0,1] = - x / denominator
        u[:,:,0,2] = 0.
        u[:,:,1,0] = x * z / denominator
        u[:,:,1,1] = y * z / denominator
        u[:,:,1,2] = - denominator
        u[:,:,2] = normal_vec
        # revision for |z| = 1, boundary case.
        pos = torch.where(abs(z ** 2 - 1) < 1e-4)[1]
        u[:,pos,0,0] = 1 / np.sqrt(2)
        u[:,pos,0,1] = - 1 / np.sqrt(2)
        u[:,pos,0,2] = 0.
        u[:,pos,1,0] = z[:,pos] / np.sqrt(2)
        u[:,pos,1,1] = z[:,pos] / np.sqrt(2)
        u[:,pos,1,2] = 0.
        u[:,pos,2,0] = 0.
        u[:,pos,2,1] = 0.
        u[:,pos,2,2] = z[:,pos]
        return u.data

    def get_original_point_cloud(self, new_points, spin_axis_matrix, translation_matrix):
        """Calculate the spin-axis matrix.

        Args:
            new_points (torch.cuda.FloatTensor): the transformed point cloud with N points, [1, N, 3].
            spin_axis_matrix (torch.cuda.FloatTensor): the rotate matrix for transformation, [1, N, 3, 3].
            translation_matrix (torch.cuda.FloatTensor): the offset matrix for transformation, [1, N, 3, 3].
        """
        inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), new_points.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
        inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
        inputs = inputs.squeeze(-1) # P, [1, N, 3]
        return inputs


    def get_transformed_point_cloud(self, points, normal_vec):
        """Calculate the spin-axis matrix.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
            normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
        """
        intercept = torch.mul(points, normal_vec).sum(-1, keepdim=True) # P \cdot N, [1, N, 1]
        spin_axis_matrix = self.get_spin_axis_matrix(normal_vec) # U, [1, N, 3, 3]
        translation_matrix = torch.mul(intercept, normal_vec).data # (P \cdot N) N, [1, N, 3]
        new_points = points + translation_matrix #  P + (P \cdot N) N, [1, N, 3]
        new_points = new_points.unsqueeze(-1) # P + (P \cdot N) N, [1, N, 3, 1]
        new_points = torch.matmul(spin_axis_matrix, new_points) # P' = U (P + (P \cdot N) N), [1, N, 3, 1]
        new_points = new_points.squeeze(-1).data # P', [1, N, 3]
        return new_points, spin_axis_matrix, translation_matrix

        
    def forward(self, x:torch.Tensor, adv_x:torch.Tensor, gt:torch.Tensor, model:nn.Module):
        
        normal_vec = points[:,:,-3:].data # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        clip_func = ClipPointsLinf(budget=self.eps)# * np.sqrt(3*1024))

        step = 0

        for i in range(self.max_steps):
            # P -> P', detach()
            new_points, spin_axis_matrix, translation_matrix = self.get_transformed_point_cloud(points, normal_vec)
            new_points = new_points.detach()

            new_points.requires_grad = True
            # P' -> P
            points = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
            points = points.transpose(1, 2) # P, [1, 3, N]
            # get white-box gradients
            if not self.defense_method is None:
                logits = self.wb_classifier(self.pre_head(points))
            else:
                logits = self.wb_classifier(points)
            loss = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
            self.wb_classifier.zero_grad()
            loss.backward()
            # print(loss.item(), logits.max(1)[1], target)
            grad = new_points.grad.data # g, [1, N, 3]
            grad[:,:,2] = 0.

            # update P', P and N
            # # Linf
            # new_points = new_points - self.step_size * torch.sign(grad)
            # L2
            norm = torch.sum(grad ** 2, dim=[1, 2]) ** 0.5
            
            new_points = new_points - self.step_size * np.sqrt(3*1024) * grad / (norm[:, None, None] + 1e-9)
            points = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix) # P, [1, N, 3]
            points = clip_func(points, ori_points)

            normal_vec = self.get_normal_vector(points) # N, [1, N, 3]

        with torch.no_grad():
            adv_points = points.data
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.transpose(1, 2).detach()))
            else:
                adv_logits = self.classifier(points.transpose(1, 2).detach())
            adv_target = adv_logits.data.max(1)[1]

        
        # if self.top5_attack:
        #     target_top_5 = adv_logits.topk(5)[1]
        #     if target in target_top_5:
        #         adv_target = target
        #     else:
        #         adv_target = -1

        del normal_vec, grad, new_points, spin_axis_matrix, translation_matrix
        return adv_points
        
        return adv_x
        
    
    

def shape_invariant_ifgm(self, points, target):
    """Black-box I-FGSM based on shape-invariant sensitivity maps.

    Args:
        points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
        target (torch.cuda.LongTensor): the label for points, [1].
    """

