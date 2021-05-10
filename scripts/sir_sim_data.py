# PyTorch stuff
import numpy as np
import torch
import torchsde
import sys

import time

# needed for torchsde
sys.setrecursionlimit(1500)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FUNCTIONS --- #


class SIR_SDE(torch.nn.Module):

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, params, N):

        super().__init__()

        # parameters
        self.params = params
        self.N = N

    # Drift vector
    def f(self, t, x):

        p_inf = self.params[:, 0] * x[:, 0] * x[:, 1] / self.N
        p_rec = self.params[:, 1] * x[:, 1]

        return torch.stack([-p_inf, p_inf - p_rec], dim=1)

    # Diffusion matrix
    def g(self, t, x):

        # Clamp data if beyond boundaries
        with torch.no_grad():
            x.clamp_(0, self.N)

        p_inf = self.params[:, 0] * x[:, 0] * x[:, 1] / self.N
        p_rec = self.params[:, 1] * x[:, 1]

        return torch.stack(
            [
                -torch.sqrt(p_inf),
                torch.zeros(len(p_inf), device=device),
                torch.sqrt(p_inf),
                -torch.sqrt(p_rec),
            ],
            dim=1,
        ).reshape(-1, 2, 2)


# --- HYPER-PARAMETERS --- #

DATASIZE = 10000  # number of time-series to simulate

N = 500.0  # fixed population size
I0 = 2.0  # initial number of infected
T0, T = 0, 100  # initial and final time
GRID = 10000  # time-grid

# --- SAMPLE FROM PRIOR --- #

# gamma prior
ss = 0.50
m_beta, m_gamma = np.log(0.50), np.log(0.10)
param_beta = np.random.normal(m_beta, ss, DATASIZE).reshape(-1, 1)  # beta
param_gamma = np.random.normal(m_gamma, ss, DATASIZE).reshape(-1, 1)  # gamma
ps = np.hstack((np.exp(param_beta), np.exp(param_gamma)))

# convert to tensor
prior_samples = torch.tensor(ps, dtype=torch.float, device=device)

# --- SOLVE THE SDEs --- #

start_time = time.time()

# define the SDE object and solve SDE equations
sde = SIR_SDE(prior_samples, N).to(device)  # sde object
ts = torch.linspace(T0, T, GRID, device=device)  # time grid
y0 = torch.tensor(DATASIZE * [[N - I0, I0]], device=device)  # starting point
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

torch.save(save_dict, "data/sir_sde_data.pt")

print("Simulation Time: %s seconds" % (end_time - start_time))
