import numpy as np
from tqdm import tqdm as tqdm
import random
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde

# PyTorch stuff
import torch

# for parallel processing
from joblib import Parallel, delayed
import multiprocessing

# MIBED stuff
# from mibed.simulators.linear import *

# ----- HYPER-PARAMETERS ----- #

DATASIZE = 2000
MC_NUM = 1000
N_JOBS = 8
NOISE_KDE_SAMPLES = 50000  # samples to be used to create noise KDE

# select the design to evaluate MI at
# d = np.array([-2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).reshape(-1, 1)
d = np.array([[10]])

# ----- FUNCTIONS ----- #

# First fit a Gaussian KDE over noise samples
eps = np.random.normal(0, 1, NOISE_KDE_SAMPLES)
nu = np.random.gamma(2, 2, NOISE_KDE_SAMPLES)
noise = eps + nu
density = gaussian_kde(noise)

def get_margdata(y, d, pp):
    
    p_data = list()
    for p in pp:
        d_data = list()
        for idx in range(d.shape[0]):
            
            pred = p[0] + p[1] * d[idx][0]
            diff = y[idx] - pred
            pdf = density(diff)[0]
            
            d_data.append(pdf)
        p_data.append(d_data)
        
    return p_data

def simulator_linear(d, prior):
    
    # sample data
    ys = list()
    for di in d:
        
        # get noise samples
        eps = np.random.normal(0, 1, prior.shape[0])
        nu = np.random.gamma(2, 2, prior.shape[0])
        noise_eps, noise_nu = 1, 1
        
        ysi = prior[:,0] + prior[:,1] * di[0] + noise_eps*eps + noise_nu*nu

        ys.append(ysi)

    ys = np.array(ys).T
    
    return ys

# ----- RUN ----- #

# Get regular prior samples
m, s = 0, 3
param_0 = np.random.normal(m, s, DATASIZE).reshape(-1,1)
param_1 = np.random.normal(m, s, DATASIZE).reshape(-1,1)
prior = np.hstack((param_0, param_1))

# simulate prior samples
ys = simulator_linear(d, prior)

# compute likelihoods
like_pdfs = list()
for idx in tqdm(range(d.shape[0])):
    ys_d = ys[:,idx]
    
    prediction = prior[:,0] + prior[:,1] * d[idx][0]
    
    diff = ys_d - prediction
    pdf = density(diff)
    
    like_pdfs.append(pdf)
like_pdfs = np.array(like_pdfs)

# compute marginals
prior_shuffle = np.random.permutation(prior[:MC_NUM])
marg_data = Parallel(n_jobs=int(N_JOBS))(delayed(get_margdata)(y, d, prior_shuffle) for y in tqdm(ys))
marg_data = np.array(marg_data)

# multiply across dimensions
like = np.prod(like_pdfs.T, axis=1)
marg = np.mean(np.prod(marg_data, axis=2), axis=1)

# ----- SAVE ------ #

save_dict = dict()
save_dict['MC_NUM'] = MC_NUM
save_dict['design'] = d.reshape(-1)
save_dict['like'] = like
save_dict['marg'] = marg
torch.save(save_dict, '../data/reference_pe_1D_MC1000_reverse.pt')