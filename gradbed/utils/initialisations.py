import numpy as np
import torch
import copy

from gradbed.networks.fullyconnected import *


def initialise_design(modelparams, device, bounds=[0, 50], d_init=None, init_type='uniform', sort=False):
    
    # Hyper-Parameters
    DIM = modelparams['DIM']
    
    # sample from a uniform distribution
    # restore to another initial state if provided
    if d_init is None:
        if init_type == 'uniform':
            d_init = np.random.uniform(bounds[0], bounds[-1], size=DIM).reshape(-1, 1)
        elif init_type == 'latin':
            ls = np.linspace(bounds[0], bounds[1], DIM + 1)
            d_init = np.array([np.random.uniform(ls[i], ls[i+1]) for i in range(DIM)]).reshape(-1, 1)
    else:
        pass
    
    if sort:
        d_init = np.sort(d_init, axis=0)
    
    # Convert to PyTorch Tensors; put on CPU/GPU
    d = torch.tensor(d_init, dtype=torch.float, device=device, requires_grad=True)
    
    return d, d_init


def initialise_model(modelparams, device, task='pe', model_init_state=None, batch_norm=False):
    
    # Hyper-Parameters
    DIM = modelparams['DIM']
    L = modelparams['L']
    H = modelparams['H']
    
    # get the right number of dimensions
    if task == 'pe':
        dim1 = 2
    elif task == 'md':
        dim1 = 1
    else:
        raise NotImplementederror('This task was not implemented.')
    dim2 = int(DIM)  # Data dimensions = Design dimensions for this model
    
    # define model
    model = FullyConnected(var1_dim=dim1, var2_dim=dim2, L=L, H=H)
    model.to(device);
    
    # extract model state (i.e. initial parameters)
    # or restore it to a specific initial state
    if model_init_state is None:
        model_init_state = copy.deepcopy(model.state_dict())
    else:
        model.load_state_dict(model_init_state)
    
    return model, model_init_state