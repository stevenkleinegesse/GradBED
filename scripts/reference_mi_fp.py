import numpy as np
from tqdm import tqdm as tqdm
import random
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from scipy import integrate

# PyTorch stuff
import torch

# for parallel processing
from joblib import Parallel, delayed
import multiprocessing

# GradBED stuff
from gradbed.simulators.linear import *

# ----- HYPER-PARAMETERS ----- #

DATASIZE = 800 # 800
N_GRID = 50
N_JOBS = 8
NOISE_KDE_SAMPLES = 50000  # samples to be used to create noise KDE

# grid to evaluate y_T on
y_T_grid = np.linspace(-50, 50, 100)

# select the design to evaluate MI at
d_fp = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).reshape(-1, 1)
future_d = np.array([[4.0]])

# ----- FUNCTIONS ----- #

# First fit a Gaussian KDE over noise samples
eps = np.random.normal(0, 1, NOISE_KDE_SAMPLES)
nu = np.random.gamma(2, 2, NOISE_KDE_SAMPLES)
noise = eps + nu
density = gaussian_kde(noise)

def compute_prior_density(thetas, m=[0,0], s=3):
    
    # compute delta
    delta = np.abs(thetas[0][0] - thetas[1][0])
    
    # create prior object
    rv = multivariate_normal(
        m, [[s ** 2, 0.0], [0.0, s ** 2]])
    
    # evaluate prior densities
    pdf = rv.pdf(thetas)
    Z = np.sum(pdf) * delta ** 2
    pdf = pdf / Z
    
    return pdf

def compute_refpost_on_grid(d, y_obs, theta_grid, tt, pdf_prior, density, bar=True):
    
    # compute likelihoods for each dimension of d
    like_pdfs = list()
    for idx in tqdm(range(d.shape[0]), disable=not bar):
        ys_d = y_obs[idx]
        prediction = theta_grid[:,0] + theta_grid[:,1] * d[idx][0]
        diff = ys_d - prediction
        pdf = density(diff)
        like_pdfs.append(pdf)
    like_pdfs = np.array(like_pdfs)

    # multiply likelihoods to get joint; assume independence across dims
    joint_like = np.prod(like_pdfs.T, axis=1)
    
    # manually normalize over grid and compute posterior
    prod = joint_like * pdf_prior
    Z = integrate.simpson(integrate.simpson(
        prod.reshape(len(tt), len(tt)), tt, axis=1), tt)
    pdf_post = prod / Z
    
    return pdf_post

def compute_pred_post(y, d, y_T_grid, future_d, theta_grid, tt, pdf_prior, density, bar=True):
    
    # compute reference posterior 
    pdf_post = compute_refpost_on_grid(
    d, y.reshape(-1, 1), theta_grid_post, 
    tt_post, pdf_prior_postgrid, density, bar=False)
    
    # compute posterior prediction
    no_noise = theta_grid[:, 0] + theta_grid[:, 1]*future_d[0][0]
    pred_post = list()
    for yt in tqdm(y_T_grid, disable=not bar):

        diff = yt - no_noise
        kde = density(diff)
        joint = kde * pdf_post
        pdf = integrate.simpson(integrate.simpson(
            joint.reshape(len(tt), len(tt)), tt, axis=1), tt)
        pred_post.append(pdf)
    pred_post = np.array(pred_post)
    
    # normalize posterior prediction
    Z_post = integrate.simpson(pred_post, y_T_grid)
    pred_post = pred_post / Z_post
    
    return pred_post

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

# ----- PRIOR RUNS ----- #

# important numbers
TMIN, TMAX = -10, 10

# create grid
tt = np.linspace(TMIN, TMAX, N_GRID)
T0, T1 = np.meshgrid(tt, tt)
theta_grid = np.vstack([T0.ravel(), T1.ravel()]).T

# get normalized prior densities on grid
pdf_prior = compute_prior_density(theta_grid)

# evaluate prior predictive
no_noise = theta_grid[:, 0] + theta_grid[:, 1]*future_d[0][0]
pred = list()
for yt in tqdm(y_T_grid):
    diff = yt - no_noise
    kde = density(diff)
    joint = kde * pdf_prior
    pdf = integrate.simpson(integrate.simpson(
        joint.reshape(len(tt), len(tt)), tt, axis=1), tt)
    pred.append(pdf)
pred = np.array(pred)

# normalize
Z = integrate.simpson(pred, y_T_grid)
pred = pred / Z

# ----- POSTERIOR RUNS ----- #

# important numbers
TMIN, TMAX = -10, 10  # <------- TODO: TOO SMALL?!

# create grid
tt_post = np.linspace(TMIN, TMAX, N_GRID)
T0, T1 = np.meshgrid(tt_post, tt_post)
theta_grid_post = np.vstack([T0.ravel(), T1.ravel()]).T

# get normalized prior densities on grid
pdf_prior_postgrid = compute_prior_density(theta_grid_post)

# Get regular prior samples
m, s = 0, 3
param_0 = np.random.normal(m, s, DATASIZE).reshape(-1,1)
param_1 = np.random.normal(m, s, DATASIZE).reshape(-1,1)
prior_tmp = np.hstack((param_0, param_1))

# simulate data
Y = simulator_linear(d_fp, prior_tmp)

# compute posterior
N_JOBS = 8
pred_post_list = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(compute_pred_post)(
    y, d_fp, y_T_grid, future_d, theta_grid_post, 
    tt_post, pdf_prior_postgrid, density, bar=False) for y in tqdm(Y)))

# ----- SAVE ------ #

save_dict = dict()
save_dict['N_GRID'] = N_GRID
save_dict['y_T_grid'] = y_T_grid
save_dict['pred_prior'] = pred
save_dict['pred_post_list'] = pred_post_list
save_dict['design'] = d_fp
save_dict['future_d'] = future_d
torch.save(save_dict, '../data/reference_fp_test_800.pt')