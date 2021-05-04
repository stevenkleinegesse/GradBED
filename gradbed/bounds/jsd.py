import numpy as np

# PyTorch stuff
import torch
import torch.nn.functional as F

# ------ MINE-F LOSS FUNCTION ------ #

def jsd_loss(x_sample, y_sample, model, device):

    # Shuffle y-data for the second expectation
    idxs = np.random.choice(
        range(len(y_sample)), size=len(y_sample), replace=False)
    # We need y_shuffle attached to the design d
    y_shuffle = y_sample[idxs]

    # Get predictions from network
    pred_joint = model(x_sample, y_sample)
    pred_marginals = model(x_sample, y_shuffle)

    # complete the individual terms
    first_term = torch.mean(-F.softplus(-pred_joint))
    second_term = torch.mean(F.softplus(pred_marginals))

    mi = first_term - second_term

    # we want to maximize the lower bound; PyTorch minimizes
    loss = - mi

    return loss
