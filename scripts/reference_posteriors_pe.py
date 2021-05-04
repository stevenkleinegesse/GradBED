import numpy as np
import torch
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed

from gradbed.simulators.linear import *

# --- FUNCTIONS --- #

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

def compute_refpost_on_grid(
    grid, d, y_obs, pdf_prior, delta, model=1, 
    kdesamples=50000, bar=True):
    
    # Fit KDE to Noise
    eps = np.random.normal(0, 1, kdesamples)
    nu = np.random.gamma(2, 2, kdesamples)
    noise = eps + nu
    density = gaussian_kde(noise)
    
    # compute likelihoods for each dimension of d
    like_pdfs = list()
    for idx in tqdm(range(d.shape[0]), disable=not bar):
        ys_d = y_obs[idx]
        prediction = grid[:,0] + grid[:,1] * d[idx][0]
        diff = ys_d - prediction
        pdf = density(diff)
        like_pdfs.append(pdf)
    like_pdfs = np.array(like_pdfs)

    # multiply likelihoods to get joint; assume independence across dims
    joint_like = np.prod(like_pdfs.T, axis=1)
    
    # manually normalize over grid and compute posterior
    prod = joint_like * pdf_prior
    Z = np.sum(prod) * delta ** 2
    pdf_post = prod / Z
    
    return pdf_post

device = torch.device('cpu')

# --- HYPER-PARAMETERS --- #

# important numbers
TMIN, TMAX = -3, 7
N_GRID = 200
N_REPEAT = 2000
N_JOBS = 48

# designs
design = np.array([-2.,  2., -2.,  2., -2.,  2., -2.,  2., -2.,  2.]).reshape(-1, 1)
#d_jsd = np.array([-2.,  2., -2.,  2., -2.,  2., -2.,  2., -2.,  2.]).reshape(-1, 1)
#d_nce = np.array([-2.,  2., -2.,  2., -2.,  2., -2.,  2., -2.,  2.]).reshape(-1, 1)

# true theta
theta_truth = np.array([[1, 2, 3, 0, 0, 0, 0]])

# datasetclass
datasetclass = LinearDatasetPE

# filename
part = 5
FILENAME = '../data/reference_post_pe_nrepeat{}_ngrid{}_part{}'.format(int(N_REPEAT), int(N_GRID), part)

# --- PREPARATIONS --- #

tt = np.linspace(TMIN, TMAX, N_GRID)
T0, T1 = np.meshgrid(tt, tt)
positions = np.vstack([T0.ravel(), T1.ravel()]).T

# deltas
delta = (TMAX - TMIN) / (N_GRID - 1.0)

# get normalized prior densities on grid
pdf_prior = compute_prior_density(positions)

# --- COMPUTATIONS --- #

# sample observation dataset
X_truth = np.vstack(N_REPEAT * [theta_truth])
d = torch.tensor(design, dtype=torch.float, device=device)
y_obs_dataset = datasetclass(d, X_truth, device)

# compute posterior densities
pdf_post = Parallel(n_jobs=N_JOBS)(
    delayed(compute_refpost_on_grid)(
        positions, design, y_obs.data.numpy(),
        pdf_prior, delta, bar=False) for _, y_obs in tqdm(y_obs_dataset))
pdf_post = np.array(pdf_post).reshape(len(y_obs_dataset), N_GRID, N_GRID)

# --- SAVE DATA --- #

save_dict = dict()
save_dict['TMIN'] = TMIN
save_dict['TMAX'] = TMAX
save_dict['N_GRID'] = N_GRID
save_dict['N_REPEAT'] = N_REPEAT
save_dict['design'] = design
save_dict['positions'] = positions
save_dict['y_obs_dataset'] = y_obs_dataset
save_dict['pdf_post_repeat'] = pdf_post

torch.save(save_dict, '{}.pt'.format(FILENAME))
