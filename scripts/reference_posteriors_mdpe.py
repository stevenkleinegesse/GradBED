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

# GradBED stuff
from gradbed.simulators.linear import *

# ----- HYPER-PARAMETERS ----- #

N_REPEAT = 2000 # 800
N_GRID = 50
N_JOBS = 64
NOISE_KDE_SAMPLES = 50000  # samples to be used to create noise KDE

# design
d = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -3.574985748855397e-05, 1.8722064396570204e-06, 5.1568484195740893e-05, 2.0, 2.0])

# ----- FUNCTIONS ----- #

# First fit a Gaussian KDE over noise samples
eps = np.random.normal(0, 1, NOISE_KDE_SAMPLES)
nu = np.random.gamma(2, 2, NOISE_KDE_SAMPLES)
noise = eps + nu
density = gaussian_kde(noise)

def model_1(d, theta):    
    return theta[:, 0] + theta[:, 1] * d + np.random.normal(0, 1, size = len(theta)) + np.random.gamma(2, 2, size = len(theta))

def model_2(d, theta):
    
    delta = 1e-4
    dabs = np.abs(d)
    if dabs < delta:
        dabs = delta
    
    return theta[:, 0] + theta[:, 1] * np.log(dabs) + np.random.normal(0, 1, size = len(theta)) + np.random.gamma(2, 2, size = len(theta))

def model_3(d, theta):
    
    return theta[:, 0] + theta[:, 1] * np.sqrt(np.abs(d)) + np.random.normal(0, 1, size = len(theta)) + np.random.gamma(2, 2, size = len(theta))

def likelihood_MD(d, y, priorsamples, kde, inner_num=100, model=1):

    # under-sample for inner sum
    idx = np.random.choice(range(len(priorsamples)), size=inner_num, replace=False)
    # get predictions depending on the model
    if model == 1:
        pred_all = priorsamples[:,0][idx] + priorsamples[:,1][idx] * d.reshape(-1, 1)
    elif model == 2:
        dabs = np.abs(d)
        delta = 1e-4
        dabs[dabs < delta] = delta
        pred_all = priorsamples[:,0][idx] + priorsamples[:,1][idx] * np.log(dabs.reshape(-1, 1))
    elif model == 3:
        pred_all = priorsamples[:,0][idx] + priorsamples[:,1][idx] * np.sqrt(np.abs(d.reshape(-1, 1)))
    else:
        raise NotImplemented()
    # compute differences to data y
    diff = y.reshape(-1, 1) - pred_all
    # compute the density based on a kde over noise
    pdfs_inner = np.array([kde(delta) for delta in diff])
    # compute the joint likelihoods across all dimensions 
    pdfs_product = np.prod(pdfs_inner, axis=0)
    # take the mean across samples from the prior to get marginal
    pdfs_marginalized = np.mean(pdfs_product)

    return pdfs_marginalized

def likelihood_MDPE_ref(d, y, priors_grid, kde, model=1):

    # get predictions depending on the model
    if model == 1:
        pred_all = priors_grid[:,0] + priors_grid[:,1] * d.reshape(-1, 1)
    elif model == 2:
        dabs = np.abs(d)
        delta = 1e-4
        dabs[dabs < delta] = delta
        pred_all = priors_grid[:,0] + priors_grid[:,1] * np.log(dabs.reshape(-1, 1))
    elif model == 3:
        pred_all = priors_grid[:,0] + priors_grid[:,1] * np.sqrt(np.abs(d.reshape(-1, 1)))
    else:
        raise NotImplemented()

    # compute differences to data y
    diff = y.reshape(-1, 1) - pred_all
    # compute the density based on a kde over noise
    pdfs_inner = np.array([kde(delta) for delta in diff])
    # compute the joint likelihoods across all dimensions 
    pdfs_product = np.prod(pdfs_inner, axis=0)

    return pdfs_product

# ----- RUN ----- #

# Create Grid
TMIN, TMAX = -7, 7
tt = np.linspace(TMIN, TMAX, N_GRID)
T0, T1 = np.meshgrid(tt, tt)
positions = np.vstack([T0.ravel(), T1.ravel()]).T

# sample real-world observations
theta_truth = np.array([[2, 3]])
X_obs = np.vstack(N_REPEAT * [theta_truth])
YO_1 = np.array([list(model_1(di, X_obs)) for di in d])
YO_2 = np.array([list(model_2(di, X_obs)) for di in d])
YO_3 = np.array([list(model_3(di, X_obs)) for di in d])

# COMPUTE LIKELIHOODS OVER GRID

# model 1 is the truth
like_true1_1 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MDPE_ref)(d, y, positions, density, model=1) for y in tqdm(YO_1.T)))
like_true1_2 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MDPE_ref)(d, y, positions, density, model=2) for y in tqdm(YO_1.T)))
like_true1_3 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MDPE_ref)(d, y, positions, density, model=3) for y in tqdm(YO_1.T)))

# model 2 is the truth
like_true2_1 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MDPE_ref)(d, y, positions, density, model=1) for y in tqdm(YO_2.T)))
like_true2_2 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MDPE_ref)(d, y, positions, density, model=2) for y in tqdm(YO_2.T)))
like_true2_3 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MDPE_ref)(d, y, positions, density, model=3) for y in tqdm(YO_2.T)))

# model 3 is the truth
like_true3_1 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MDPE_ref)(d, y, positions, density, model=1) for y in tqdm(YO_3.T)))
like_true3_2 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MDPE_ref)(d, y, positions, density, model=2) for y in tqdm(YO_3.T)))
like_true3_3 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MDPE_ref)(d, y, positions, density, model=3) for y in tqdm(YO_3.T)))

# ----- SAVE ------ #

save_dict = dict()
save_dict['like_true1_1'] = like_true1_1
save_dict['like_true1_2'] = like_true1_2
save_dict['like_true1_3'] = like_true1_3
save_dict['like_true2_1'] = like_true2_1
save_dict['like_true2_2'] = like_true2_2
save_dict['like_true2_3'] = like_true2_3
save_dict['like_true3_1'] = like_true3_1
save_dict['like_true3_2'] = like_true3_2
save_dict['like_true3_3'] = like_true3_3
save_dict['design'] = d
save_dict['N_GRID'] = N_GRID
save_dict['TMIN'] = TMIN
save_dict['TMAX'] = TMAX
save_dict['positions'] = positions
save_dict['y_obs_true1'] = YO_1
save_dict['y_obs_true2'] = YO_2
save_dict['y_obs_true3'] = YO_3

part=4
torch.save(save_dict, '../data/reference_post_mdpe_test_part{}.pt'.format(part))
