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
        weight_decay = optimizer_conf['Adam'].get('weight_decay', 0)
        amsgrad = optimizer_conf['Adam'].get('amsgrad', False)
        betas = optimizer_conf['Adam'].get('betas', (0.9, 0.999))
        eps = optimizer_conf['Adam'].get('eps', 1e-8)
        print('optimizer:', optimizer_conf['optimizer_choice'], 'lr:', lr, 
              'weight_decay:', weight_decay, 'amsgrad:', amsgrad)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, 
                              amsgrad=amsgrad, betas=betas, eps=eps)
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
    
    