# -*- coding: utf-8 -*-

"""
# optimizers.py
"""

import torch
import omegaconf
from omegaconf import OmegaConf

def get_optimizer(params, conf):
    optimizer_conf = conf['optimizer']

    optimizer_choice = optimizer_conf['optimizer_choice']

    if optimizer_choice == 'Adam':
        lr = optimizer_conf['Adam']['lr']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.Adam(params, lr)
    elif optimizer_choice == 'AdamW':
        lr = optimizer_conf['AdamW']['lr']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.AdamW(params, lr)
    elif optimizer_choice == 'SGD':
        lr = lr = optimizer_conf['SGD']['lr']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.SGD(params, lr)
    elif optimizer_choice == 'RMSprop':
        lr = optimizer_conf['RMSprop']['lr']
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr)
        return torch.optim.RMSprop(params, lr)
    else:
        raise Exception('optimizer', optimizer_conf['optimizer_choice'] ,'not implemented.')
    
    