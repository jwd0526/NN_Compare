# -*- coding: utf-8 -*-

"""
# schedulers.py
"""

import torch
import omegaconf
from omegaconf import OmegaConf


def get_scheduler(optimizer, conf):
    scheduler_conf = conf['scheduler']
    scheduler_choice = scheduler_conf['scheduler_choice']

    if scheduler_choice == 'MultiStepLR':
        milesones = list(scheduler_conf[scheduler_choice]['milestones'])
        print('scheduler:', scheduler_choice, 'milesones:', milesones)
        if 'gamma' in scheduler_conf[scheduler_choice]:
            gamma = scheduler_conf[scheduler_choice]['gamma']
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milesones, gamma)
        else:
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milesones)

    elif scheduler_choice == 'CosineAnnealingWarmRestarts':
        T_0 = scheduler_conf[scheduler_choice]['T_0']
        T_mult = scheduler_conf[scheduler_choice].get('T_mult', 1)
        eta_min = scheduler_conf[scheduler_choice].get('eta_min', 0)
        print('scheduler:', scheduler_choice, 'T_0:', T_0, 'T_mult:', T_mult, 'eta_min:', eta_min)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mult, eta_min=eta_min)

    elif scheduler_choice == 'CyclicLR':
        base_lr = scheduler_conf[scheduler_choice]['base_lr']
        max_lr = scheduler_conf[scheduler_choice]['max_lr']
        step_size_up = scheduler_conf[scheduler_choice]['step_size_up']
        print('scheduler:', scheduler_conf['scheduler_choice'], 'base_lr:', base_lr, 
        'max_lr:', max_lr, 'step_size_up:', step_size_up)
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up)
    
    elif scheduler_choice == 'OneCycleLR':
        max_lr = scheduler_conf[scheduler_choice]['max_lr']
        total_steps = conf['hyperparameters'].get('epoch', 100)  # Default to 100 if not specified
        
        # Get optional parameters with defaults
        pct_start = scheduler_conf[scheduler_choice].get('pct_start', 0.3)
        div_factor = scheduler_conf[scheduler_choice].get('div_factor', 25.0)
        final_div_factor = scheduler_conf[scheduler_choice].get('final_div_factor', 10000.0)
        
        print('scheduler:', scheduler_choice, 
              'max_lr:', max_lr, 
              'total_steps:', total_steps,
              'pct_start:', pct_start)
              
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
        
    elif scheduler_choice == 'none':
        return None
    else:
        raise Exception('scheduler', scheduler_conf['scheduler_choice'] ,'not implemented.')