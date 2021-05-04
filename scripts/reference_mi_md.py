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

DATASIZE = 3000 # 200 #2000 # 3000
MC_NUM = 1000
N_JOBS = 64
NOISE_KDE_SAMPLES = 50000  # samples to be used to create noise KDE

# design
good = True
if good:  # global optimum
    #d = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, 6.50523288641125e-05, 2.0, 2.0, 2.0, 2.0])
    d = np.array([-2.0, -2.0, -2.0, -8.075553341768682e-05, -4.574532067636028e-05, 5.260134639684111e-05, 2.0, 2.0, 2.0, 2.0])
else:  # local optimum
    d = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

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

# ----- RUN ----- #

# sample from prior for each model
prior_1 = multivariate_normal(mean=np.zeros(2), cov = np.array([[3 ** 2, 0], [0, 3 ** 2]]))
prior_2 = multivariate_normal(mean=np.zeros(2), cov = np.array([[3 ** 2, 0], [0, 3 ** 2]]))
prior_3 = multivariate_normal(mean=np.zeros(2), cov = np.array([[3 ** 2, 0], [0, 3 ** 2]]))
p1_samples = prior_1.rvs(size=DATASIZE)
p2_samples = prior_2.rvs(size=DATASIZE)
p3_samples = prior_3.rvs(size=DATASIZE)

# sample from the prior predictive
y_1 = np.array([list(model_1(di, p1_samples)) for di in d])
y_2 = np.array([list(model_2(di, p2_samples)) for di in d])
y_3 = np.array([list(model_3(di, p3_samples)) for di in d])

# compute diagonal terms
like_1 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MD)(d, y, p1_samples, density, MC_NUM, model=1) for y in tqdm(y_1.T)))
like_2 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MD)(d, y, p2_samples, density, MC_NUM, model=2) for y in tqdm(y_2.T)))
like_3 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MD)(d, y, p3_samples, density, MC_NUM, model=3) for y in tqdm(y_3.T)))

# compute cross-terms
like_1in2 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MD)(d, y, p2_samples, density, MC_NUM, model=2) for y in tqdm(y_1.T)))
like_1in3 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MD)(d, y, p3_samples, density, MC_NUM, model=3) for y in tqdm(y_1.T)))
like_2in1 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MD)(d, y, p1_samples, density, MC_NUM, model=1) for y in tqdm(y_2.T)))
like_2in3 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MD)(d, y, p3_samples, density, MC_NUM, model=3) for y in tqdm(y_2.T)))
like_3in1 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MD)(d, y, p1_samples, density, MC_NUM, model=1) for y in tqdm(y_3.T)))
like_3in2 = np.array(Parallel(n_jobs=int(N_JOBS))(delayed(likelihood_MD)(d, y, p2_samples, density, MC_NUM, model=2) for y in tqdm(y_3.T)))

# ----- COMPUTATIONS ----- #

# sometimes the probability for the second model is zero (because of the cap)
idx_good = [i for i in range(len(like_2)) if like_2[i]!=0]

# compute marginal probabilities
m1 = (like_1 + like_1in2 + like_1in3) / 3
m2 = (like_2 + like_2in1 + like_2in3) / 3
m3 = (like_3 + like_3in1 + like_3in1) / 3

# compute log probabilities; entropy term 'l' and cross-entropy term 'm'
l = (np.log(like_1[idx_good]) + np.log(like_2[idx_good]) + np.log(like_3[idx_good])) / 3
m = (np.log(m1[idx_good]) + np.log(m2[idx_good]) + np.log(m3[idx_good])) / 3

# log ratio
mi_int = l - m

# ----- SAVE ------ #

save_dict = dict()
save_dict['like_1'] = like_1
save_dict['like_2'] = like_2
save_dict['like_3'] = like_3
save_dict['like_1in2'] = like_1in2
save_dict['like_1in3'] = like_1in3
save_dict['like_2in1'] = like_2in1
save_dict['like_2in3'] = like_2in3
save_dict['like_3in1'] = like_3in1
save_dict['like_3in2'] = like_3in2
save_dict['design'] = d
save_dict['MC_NUM'] = MC_NUM
save_dict['mi_estimate'] = np.mean(mi_int)
save_dict['mi_error'] = np.std(mi_int) / np.sqrt(len(mi_int))

if good:
    torch.save(save_dict, '../data/reference_md_new.pt')
else:
    torch.save(save_dict, '../data/reference_md_sub_test.pt')
