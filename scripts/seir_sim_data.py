# PyTorch stuff
import numpy as np
import torch
import torchsde
import sys

import time

# Needed for torchsde
sys.setrecursionlimit(1500)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FUNCTIONS --- #


class SEIR_SDE(torch.nn.Module):

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, params, N):

        super().__init__()

        # parameters
        self.params = params
        self.N = N

    # Drift vector
    def f(self, t, x):

        p_inf = self.params[:, 0] * x[:, 0] * x[:, 2] / self.N
        p_exp = self.params[:, 1] * x[:, 1]
        p_rec = self.params[:, 2] * x[:, 2]

        return torch.stack([-p_inf, p_inf - p_exp, p_exp - p_rec], dim=1)

    # Diffusion matrix
    def g(self, t, x):

        # Clamp data if beyond boundaries
        with torch.no_grad():
            x.clamp_(0, self.N)

        p_inf = self.params[:, 0] * x[:, 0] * x[:, 2] / self.N
        p_exp = self.params[:, 1] * x[:, 1]
        p_rec = self.params[:, 2] * x[:, 2]

        G = torch.stack(
            [
                -torch.sqrt(p_inf),
                torch.zeros(len(p_inf), device=device),
                torch.zeros(len(p_inf), device=device),
                torch.sqrt(p_inf),
                -torch.sqrt(p_exp),
                torch.zeros(len(p_inf), device=device),
                torch.zeros(len(p_inf), device=device),
                torch.sqrt(p_exp),
                torch.sqrt(p_rec)
            ],
            dim=1).reshape(-1, 3, 3)

        return G


# --- HYPER-PARAMETERS --- #

DATASIZE = 30000  # number of samples

N = 500.0  # fixed population size
E0 = 0  # initial number of infected / exposed
I0 = 2  # initial number of infectious
T0, T = 0, 100  # initial and final time
GRID = 10000  # time-grid

# --- SAMPLE FROM PRIOR --- #

# gamma prior
ss_beta, ss_sigma, ss_gamma = 0.50, 0.50, 0.50
m_beta, m_sigma, m_gamma = np.log(0.50), np.log(0.20), np.log(0.10)
p_beta = np.random.normal(m_beta, ss_beta, DATASIZE).reshape(-1, 1)  # beta
p_sigma = np.random.normal(m_sigma, ss_sigma, DATASIZE).reshape(-1, 1)  # sigma
p_gamma = np.random.normal(m_gamma, ss_gamma, DATASIZE).reshape(-1, 1)  # gamma
ps = np.hstack((np.exp(p_beta), np.exp(p_sigma), np.exp(p_gamma)))

# conver to tensor
prior_samples = torch.tensor(ps, dtype=torch.float, device=device)

# --- SOLVE THE SDEs --- #

start_time = time.time()

# define the SDE object and solve SDE equations
sde = SEIR_SDE(prior_samples, N).to(device)  # sde object
ts = torch.linspace(T0, T, GRID, device=device)  # time grid
y0 = torch.tensor(DATASIZE * [[N - I0 - E0, E0, I0]], device=device)  # start
ys = torchsde.sdeint(sde, y0, ts)  # solved sde

# compute gradients via finite-difference methods
grads = (ys[1:, :, :] - ys[:-1, :, :]) / (ts[1] - ts[0])

end_time = time.time()

# --- SAVE DATA --- #

save_dict = dict()
save_dict["prior_samples"] = prior_samples
save_dict["ts"] = ts
save_dict["ys"] = ys
save_dict["grads"] = grads
save_dict["N"] = N
save_dict["I0"] = I0

torch.save(save_dict, '../data/seir_sde_data.pt')

print("Simulation Time: %s seconds" % (end_time - start_time))
