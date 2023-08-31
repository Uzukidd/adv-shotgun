import os
import torch
from torch import nn

def load_models(classifier:nn.Module, checkpoint_path:str):
    """Load white-box surrogate model and black-box target model.
    """
    
    checkpoint = torch.load(checkpoint_path)

    try:
        if 'model_state_dict' in checkpoint:
            classifier.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            classifier.load_state_dict(checkpoint['model_state'])
        elif 'state_dict' in checkpoint:
            classifier.load_state_dict(checkpoint['state_dict'])
        else:
            classifier.load_state_dict(checkpoint)
    except:
        classifier = nn.DataParallel(classifier)
        classifier.load_state_dict(checkpoint)
    return classifier