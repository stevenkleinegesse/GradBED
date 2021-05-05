import numpy as np

# PyTorch stuff
import torch
import torch.utils.data
from torch.distributions import gamma

# ------ FUNCTIONS ------ #

def sim_linear_data(d, prior, device):

    # sample random normal noise
    n_n = torch.empty(
        (len(d), len(prior)),
        device=device,
        dtype=torch.float).normal_(mean=0, std=1)

    # sample random gamma noise
    n_g = gamma.Gamma(
        torch.tensor([2.0], device=device),
        torch.tensor([1 / 2.0], device=device)).sample(
            sample_shape=(len(d), len(prior))).reshape(len(d), len(prior))

    # prepare mask
    vz = torch.zeros_like(n_n, device=device)

    # predictions of model 1 + masking
    y_1 = prior[:, 1] + torch.mul(prior[:, 2], d) + torch.where(
        prior[:, 0] == 1, n_n + n_g, vz)

    # predictions of model 2
    delta = torch.tensor(1e-4, dtype=torch.float, device=device)
    d_mask_abs = torch.where(torch.abs(d) > delta, torch.abs(d), delta)
    y_2 = prior[:, 3] + torch.mul(prior[:, 4], torch.log(
        d_mask_abs)) + torch.where(prior[:, 0] == 2, n_n + n_g, vz)

    # predictions of model 3
    y_3 = prior[:, 5] + torch.mul(prior[:, 6], torch.sqrt(
        torch.abs(d))) + torch.where(prior[:, 0] == 3, n_n + n_g, vz)

    data = y_1.T + y_2.T + y_3.T

    return data


def sim_linear_prior(batch_size, model=None):

    # mean and standard deviation for all Gaussian priors
    m, s = 0, 3

    # sample params for each model
    param_11 = np.random.normal(m, s, batch_size).reshape(-1, 1)
    param_12 = np.random.normal(m, s, batch_size).reshape(-1, 1)
    param_21 = np.random.normal(m, s, batch_size).reshape(-1, 1)
    param_22 = np.random.normal(m, s, batch_size).reshape(-1, 1)
    param_31 = np.random.normal(m, s, batch_size).reshape(-1, 1)
    param_32 = np.random.normal(m, s, batch_size).reshape(-1, 1)

    if model is None:
        # sample model indicators
        param_m = np.random.choice([1, 2, 3], batch_size).reshape(-1, 1)

    elif (model == 1) or (model == 2) or (model == 3):
        # model parameter is constant
        param_m = np.ones(shape=(batch_size, 1)) * model

    else:
        raise NotImplementedError('This model is not implemented.')

    # stack parameters
    prior = np.hstack((
        param_m, param_11, param_12, param_21, param_22, param_31, param_32))

    # produce masked array
    for p in prior:
        if p[0] == 1:
            p[3:] = 0.0
        elif p[0] == 2:
            p[1:3] = 0.0
            p[-2:] = 0.0
        elif p[0] == 3:
            p[1:-2] = 0.0

    return prior


class LinearDatasetMD(torch.utils.data.Dataset):

    def __init__(self, d, prior, device):

        """
        A linear toy model dataset for model discrimination.
        Parameters
        ----------
        designs: torch.tensor
            Design variables that we want to optimise.
        prior: numpy array
            Samples from the prior distribution.
        device: torch.device
            Device to run the training process on.
        """
        super(LinearDatasetMD, self).__init__()

        # convert designs and prior samples to PyTorch tensors
        X = torch.tensor(
            prior, dtype=torch.float, device=device, requires_grad=False)
        self.m = X[:, 0].reshape(-1, 1)

        # simulate data
        self.Y = sim_linear_data(d, X, device)

    def __getitem__(self, idx):
        """ Get Prior samples and data by index.
        Parameters
        ----------
        idx : int
            Item index
        Returns
        -------
        Batched prior samples, batched data samples
        """
        return self.m[idx], self.Y[idx]

    def __len__(self):
        """Number of samples in the dataset"""
        return len(self.m)


class LinearDatasetMDPE(torch.utils.data.Dataset):

    def __init__(self, d, prior, device):

        """
        A linear toy model dataset for model discrimination and parameter
        estimation.

        Parameters
        ----------
        designs: torch.tensor
            Design variables that we want to optimise.
        prior: numpy array
            Samples from the prior distribution.
        device: torch.device
            Device to run the training process on.
        """
        super(LinearDatasetMDPE, self).__init__()

        # convert designs and prior samples to PyTorch tensors
        self.X = torch.tensor(
            prior, dtype=torch.float, device=device, requires_grad=False)

        # simulate data
        self.Y = sim_linear_data(d, self.X, device)

    def __getitem__(self, idx):
        """ Get Prior samples and data by index.
        Parameters
        ----------
        idx : int
            Item index
        Returns
        -------
        Batched prior samples, batched data samples
        """
        return self.X[idx], self.Y[idx]

    def __len__(self):
        """Number of samples in the dataset"""
        return len(self.X)


class LinearDatasetPE(torch.utils.data.Dataset):

    def __init__(self, d, prior, device):

        """
        A linear toy model dataset for parameter estimation.
        Parameters
        ----------
        designs: torch.tensor
            Design variables that we want to optimise.
        prior: numpy array
            Samples from the prior distribution.
        device: torch.device
            Device to run the training process on.
        """
        super(LinearDatasetPE, self).__init__()

        # convert designs and prior samples to PyTorch tensors
        X = torch.tensor(
            prior, dtype=torch.float, device=device, requires_grad=False)

        # all elements of model indicator prior need to be the same
        indicator = X[:, 0][0]
        assert torch.all(torch.eq(X[:, 0], indicator))

        # check for which model we want to estimate the parameters
        if indicator == 1:
            self.X = X[:, 1:3]
        elif indicator == 2:
            self.X = X[:, 3:-2]
        elif indicator == 3:
            self.X = X[:, -2:]
        else:
            raise NotImplementedError('This model is not implemented.')

        # simulate data
        self.Y = sim_linear_data(d, X, device)

    def __getitem__(self, idx):
        """ Get Prior samples and data by index.
        Parameters
        ----------
        idx : int
            Item index
        Returns
        -------
        Batched prior samples, batched data samples
        """
        return self.X[idx], self.Y[idx]

    def __len__(self):
        """Number of samples in the dataset"""
        return len(self.X)

class LinearDatasetFP(torch.utils.data.Dataset):

    def __init__(self, d, future_d, prior, device):

        """
        A linear toy model dataset for parameter estimation.
        Parameters
        ----------
        designs: torch.tensor
            Design variables that we want to optimise.
        prior: numpy array
            Samples from the prior distribution.
        device: torch.device
            Device to run the training process on.
        """
        super(LinearDatasetFP, self).__init__()

        # convert designs and prior samples to PyTorch tensors
        pp = torch.tensor(
            prior, dtype=torch.float, device=device, requires_grad=False)

        # all elements of model indicator prior need to be the same
        indicator = pp[:, 0][0]
        assert torch.all(torch.eq(pp[:, 0], indicator))

        # simulate future data
        self.X = sim_linear_data(future_d, pp, device)

        # simulate data
        self.Y = sim_linear_data(d, pp, device)

    def __getitem__(self, idx):
        """ Get Prior samples and data by index.
        Parameters
        ----------
        idx : int
            Item index
        Returns
        -------
        Batched prior samples, batched data samples
        """
        return self.X[idx], self.Y[idx]

    def __len__(self):
        """Number of samples in the dataset"""
        return len(self.X)

# ----# ANALYTIC BACK-PROPAGATION ----- #


class LinearSimulatorAnalytic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, params, device):

        # sample random normal noise
        n_n = torch.empty(
            (len(input), len(params)),
            device=device,
            dtype=torch.float).normal_(mean=0, std=1)

        # sample random gamma noise
        n_g = gamma.Gamma(
            torch.tensor([2.0], device=device),
            torch.tensor([1 / 2.0], device=device)).sample(
                sample_shape=(len(input), len(params))).reshape(
                len(input), len(params))

        # perform forward pass
        y = (params[:, 0] + torch.mul(params[:, 1], input) + n_n + n_g).T

        ctx.save_for_backward(input, params)
        ctx.device = device

        return y

    @staticmethod
    def backward(ctx, grad_output):

        # unpack saved tensors
        input, params = ctx.saved_tensors
        device = ctx.device

        # compute the Jacobian
        y_grad = torch.eye(
            len(input),
            device=device,
            dtype=torch.float).reshape(1, len(input), len(input))
        Jac = torch.mul(
            y_grad.repeat(len(params), 1, 1), params[:, 1, None, None])

        # compute the Jacobian vector product
        grad_input = Jac.matmul(grad_output[:, :, None])

        return grad_input, None, None
