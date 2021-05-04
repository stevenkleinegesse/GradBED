# PyTorch stuff
import torch

# ------ MINE-F LOSS FUNCTION ------ #

def nce_loss(x_sample, y_sample, model, device, mode='likelihood'):

    if mode == 'likelihood':
        sum_dim = 0
    elif mode == 'posterior':
        sum_dim = 1
    else:
        raise NotImplemented("Choose either 'likelihood' or 'posterior'.")

    batch_size = x_sample.size(0)
    # Tile all possible combinations of x and y
    x_stacked = torch.stack(
        [x_sample] * batch_size, dim=0).reshape(batch_size * batch_size, -1)
    y_stacked = torch.stack(
        [y_sample] * batch_size, dim=1).reshape(batch_size * batch_size, -1)

    # get model predictions for joint data
    pred = model(x_stacked, y_stacked).reshape(batch_size, batch_size).T

    # rows of pred correspond to values of x
    # columns of pred correspond to values of y

    # log batch_size
    logK = torch.log(torch.tensor(pred.size(0), dtype=torch.float))

    # compute MI
    kl = torch.diag(pred) - torch.logsumexp(pred, dim=sum_dim) + logK
    mi = torch.mean(kl)

    # we want to maximise the mutual information
    loss = -mi

    return loss
