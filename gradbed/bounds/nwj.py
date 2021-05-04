import numpy as np

# PyTorch stuff
import torch

# ------ MINE-F LOSS FUNCTION ------ #

def nwj_loss(x_sample, y_sample, model, device):

    # Shuffle y-data for the second expectation
    idxs = np.random.choice(
        range(len(y_sample)), size=len(y_sample), replace=False)
    # We need y_shuffle attached to the design d
    y_shuffle = y_sample[idxs]

    # Get predictions from network
    pred_joint = model(x_sample, y_sample)
    pred_marginals = model(x_sample, y_shuffle)

    # Compute the MINE-f (or NWJ) lower bound
    Z = torch.tensor(np.exp(1), device=device, dtype=torch.float)
    mi_ma = torch.mean(pred_joint) - torch.mean(
        torch.exp(pred_marginals) / Z + torch.log(Z) - 1)

    # we want to maximize the lower bound; PyTorch minimizes
    loss = - mi_ma

    return loss
