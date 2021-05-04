import numpy as np
import torch
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed

# PyTorch stuff
import torch

# for parallel processing
from joblib import Parallel, delayed
import multiprocessing

# GradBED stuff
from gradbed.simulators.linear import *

# --- HYPER-PARAMETERS ---- #

N_REPEAT = 5000
MC_NUM = 1000
DATASIZE = MC_NUM * 10
N_JOBS = 64
NOISE_KDE_SAMPLES = 50000  # samples to be used to create noise KDE

# design
good = True
if good:  # global optimum
    #d = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, 6.50523288641125e-05, 2.0, 2.0, 2.0, 2.0])
    d = np.array([-2.0, -2.0, -2.0, -8.075553341768682e-05, -4.574532067636028e-05, 5.260134639684111e-05, 2.0, 2.0, 2.0, 2.0])
else:  # local optimum
    d = np.array([-2.,  2.,  2.,  2.,  2., -2., -2., -2.,  2., -2.])

# --- FUNCTIONS --- #

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

device = torch.device('cpu')

# --- RUN --- #

# sample real-world observations
theta_truth = np.array([[2, 3]])
X_obs = np.vstack(N_REPEAT * [theta_truth])
y_obs_true_1 = np.array([list(model_1(di, X_obs)) for di in d])
y_obs_true_2 = np.array([list(model_2(di, X_obs)) for di in d])
y_obs_true_3 = np.array([list(model_3(di, X_obs)) for di in d])

# sample from prior for each model
prior_1 = multivariate_normal(mean=np.zeros(2), cov = np.array([[3 ** 2, 0], [0, 3 ** 2]]))
prior_2 = multivariate_normal(mean=np.zeros(2), cov = np.array([[3 ** 2, 0], [0, 3 ** 2]]))
prior_3 = multivariate_normal(mean=np.zeros(2), cov = np.array([[3 ** 2, 0], [0, 3 ** 2]]))
p1_samples = prior_1.rvs(size=DATASIZE)
p2_samples = prior_2.rvs(size=DATASIZE)
p3_samples = prior_3.rvs(size=DATASIZE)

# compute diagonal terms
like_true1_1 = np.array(Parallel(n_jobs=int(N_JOBS))(
    delayed(likelihood_MD)(d, y_obs, p1_samples, density, MC_NUM, model=1) for y_obs in tqdm(y_obs_true_1.T)))
like_true2_2 = np.array(Parallel(n_jobs=int(N_JOBS))(
    delayed(likelihood_MD)(d, y_obs, p2_samples, density, MC_NUM, model=2) for y_obs in tqdm(y_obs_true_2.T)))
like_true3_3 = np.array(Parallel(n_jobs=int(N_JOBS))(
    delayed(likelihood_MD)(d, y_obs, p3_samples, density, MC_NUM, model=3) for y_obs in tqdm(y_obs_true_3.T)))

# compute cross-terms
like_true1_2 = np.array(Parallel(n_jobs=int(N_JOBS))(
    delayed(likelihood_MD)(d, y_obs, p2_samples, density, MC_NUM, model=2) for y_obs in tqdm(y_obs_true_1.T)))
like_true1_3 = np.array(Parallel(n_jobs=int(N_JOBS))(
    delayed(likelihood_MD)(d, y_obs, p3_samples, density, MC_NUM, model=3) for y_obs in tqdm(y_obs_true_1.T)))

like_true2_1 = np.array(Parallel(n_jobs=int(N_JOBS))(
    delayed(likelihood_MD)(d, y_obs, p1_samples, density, MC_NUM, model=1) for y_obs in tqdm(y_obs_true_2.T)))
like_true2_3 = np.array(Parallel(n_jobs=int(N_JOBS))(
    delayed(likelihood_MD)(d, y_obs, p3_samples, density, MC_NUM, model=3) for y_obs in tqdm(y_obs_true_2.T)))

like_true3_1 = np.array(Parallel(n_jobs=int(N_JOBS))(
    delayed(likelihood_MD)(d, y_obs, p1_samples, density, MC_NUM, model=1) for y_obs in tqdm(y_obs_true_3.T)))
like_true3_2 = np.array(Parallel(n_jobs=int(N_JOBS))(
    delayed(likelihood_MD)(d, y_obs, p2_samples, density, MC_NUM, model=2) for y_obs in tqdm(y_obs_true_3.T)))

# filename
part=1
if good:
    FILENAME = '../data/reference_post_md_new_part{}'.format(part)
else:
    FILENAME = '../data/reference_post_md_sub_part{}'.format(part)

# --- SAVE DATA --- #

save_dict = dict()
save_dict['MC_NUM'] = MC_NUM
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
save_dict['y_obs_true_1'] = y_obs_true_1
save_dict['y_obs_true_2'] = y_obs_true_2
save_dict['y_obs_true_3'] = y_obs_true_3

torch.save(save_dict, '{}.pt'.format(FILENAME))
