# PyTorch stuff
import numpy as np
import torch
import torchsde
import sys

# somehow I need this for torchsde
sys.setrecursionlimit(1500)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- FUNCTIONS --- #

class SEIR_SDE(torch.nn.Module):
    
    noise_type = 'general'
    sde_type = 'ito'
    
    def __init__(self, params, N):
        
        super().__init__()
        
        # parameters
        self.params = params
        self.N = N

    # Drift
    def f(self, t, x):
        
        p_inf = self.params[:, 0] * x[:, 0] * x[:, 2] / self.N
        p_exp = self.params[:, 1] * x[:, 1]
        p_rec = self.params[:, 2] * x[:, 2]
        
        return torch.stack([-p_inf, p_inf - p_exp, p_exp - p_rec], axis=1)
    
    # Diffusion
    def g(self, t, x):
        
        # Clamp data if beyond boundaries
        with torch.no_grad():
            x.clamp_(0, self.N)
        
        p_inf = self.params[:, 0] * x[:, 0] * x[:, 2] / self.N
        p_exp = self.params[:, 1] * x[:, 1]
        p_rec = self.params[:, 2] * x[:, 2]
        
        G = torch.stack([-torch.sqrt(p_inf), torch.zeros(len(p_inf)), torch.zeros(len(p_inf)), 
                         torch.sqrt(p_inf), -torch.sqrt(p_exp), torch.zeros(len(p_inf)),
                         torch.zeros(len(p_inf)), torch.sqrt(p_exp), torch.sqrt(p_rec)], axis=1)
        
        return G.reshape(-1, 3, 3)
    
# --- HYPER-PARAMETERS --- #

DATASIZE = 5000  # number of samples

N = 500.0  # fixed population size
E0 = 0
I0 = 2  # initial number of infected
T0, T = 0, 100  # initial and final time
GRID = 10000  # time-grid

# --- SAMPLE FROM PRIOR --- #

# gamma prior
ss_beta, ss_sigma, ss_gamma = 0.50, 0.50, 0.50
m_beta, m_sigma, m_gamma = np.log(0.50), np.log(0.20), np.log(0.10)
param_beta = np.random.normal(m_beta, ss_beta, DATASIZE).reshape(-1, 1)  # beta
param_sigma = np.random.normal(m_sigma, ss_sigma, DATASIZE).reshape(-1, 1)  # sigma
param_gamma = np.random.normal(m_gamma, ss_gamma, DATASIZE).reshape(-1, 1)  # gamma
ps = np.hstack((np.exp(param_beta), np.exp(param_sigma), np.exp(param_gamma)))

# # uniform prior
# beta_mean, gamma_mean = 0.50, 0.10
# beta_range, gamma_range = 0.10, 0.10
# param_beta = np.random.uniform(beta_mean-beta_range, beta_mean+beta_range, DATASIZE).reshape(-1, 1)  # beta
# param_gamma = np.random.uniform(gamma_mean-gamma_range, gamma_mean+gamma_range, DATASIZE).reshape(-1, 1)  # beta
# ps = np.hstack((param_beta, param_gamma))

prior_samples = torch.tensor(ps, dtype=torch.float)

# --- SOLVE THE SDEs --- #

sde = SEIR_SDE(prior_samples, N)  # sde object
ts = torch.linspace(T0, T, GRID)  # time grid
y0 = torch.tensor(DATASIZE * [[N - I0 - E0, E0, I0]])  # initial starting points
ys = torchsde.sdeint(sde, y0, ts)  # solved sde

# compute gradients via finite-difference methods
grads = (ys[1:,:,:] - ys[:-1,:,:]) / (ts[1] - ts[0]) 

# --- SAVE DATA --- #

save_dict = dict()
# save_dict['SDE_obj'] = sde
save_dict['prior_samples'] = prior_samples
save_dict['ts'] = ts
save_dict['ys'] = ys
save_dict['grads'] = grads
save_dict['N'] = N
save_dict['I0'] = I0

# torch.save(save_dict, '../data/seir_sde_data_small.pt')
torch.save(save_dict, '../data/seir_sde_data_valid.pt')