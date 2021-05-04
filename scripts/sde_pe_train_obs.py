import numpy as np
from tqdm import tqdm as tqdm
import random
import sys, os

# PyTorch stuff
import torch
import torchsde
from torch.optim.lr_scheduler import StepLR

# GradBED stuff
from gradbed.networks.fullyconnected import *
from gradbed.bounds.jsd import *
from gradbed.bounds.nwj import *
from gradbed.utils.initialisations import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Classes and Functions --- #

class SIR_SDE(torch.nn.Module):
    
    noise_type = 'general'
    sde_type = 'ito'
    
    def __init__(self, params, N):
        
        super().__init__()
        
        # parameters
        self.params = params
        self.N = N

    # Drift
    def f(self, t, x):
        
        p_inf = self.params[:, 0] * x[:, 0] * x[:, 1] / self.N
        p_rec = self.params[:, 1] * x[:, 1]
        
        return torch.stack([-p_inf, p_inf - p_rec], axis=1)
    
    # Diffusion
    def g(self, t, x):
        
        # Clamp data if beyond boundaries
        with torch.no_grad():
            x.clamp_(0, self.N)
        
        p_inf = self.params[:, 0] * x[:, 0] * x[:, 1] / self.N
        p_rec = self.params[:, 1] * x[:, 1]
        
        return torch.stack([-torch.sqrt(p_inf),
                            torch.zeros(len(p_inf)),
                            torch.sqrt(p_inf),
                            -torch.sqrt(p_rec)],
                           axis=1).reshape(-1, 2, 2)

class SIR_SDE_Observations_Simulator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, device):
        
        # observation factor
        phi = torch.tensor(0.95, dtype=torch.float, device=device)
        
        with torch.no_grad():
            
            # compute nearest neighbours in time grid
            indices = torch.min(torch.abs(input - data['ts'][:-1]), axis=1).indices
            
            # extract number of infected from data
            y = data['ys'][:-1][indices,:,1].T
            
            # sample observations
            y_obs = torch.poisson(phi * torch.nn.ReLU()(y))
            
            # compute ratios
            delta = torch.tensor(1e-8, dtype=torch.float, device=device)
            tmp_ratios = y / (y_obs + delta)
            zer = torch.zeros_like(y_obs, dtype=torch.float, device=device)
            ratios = torch.where(y_obs == 0, zer, tmp_ratios)

        ctx.save_for_backward(input, ratios)
        ctx.device = device
        ctx.indices = indices
        ctx.phi = phi

        return y_obs

    @staticmethod
    def backward(ctx, grad_output):

        # unpack saved tensors
        input, ratios = ctx.saved_tensors
        device = ctx.device
        indices = ctx.indices
        phi = ctx.phi
        
        # extract gradients of infected from data
        y_grads = data['grads'][indices,:,1].T  # GLOBAL VARIABLE DATA
        
        # compute observational gradients
        obs_grads = (ratios - phi) * y_grads

        # compute the Jacobian
        identity = torch.eye(
            len(indices),
            device=device,
            dtype=torch.float).reshape(1, len(indices), len(indices))
        Jac = torch.mul(identity.repeat(len(obs_grads), 1, 1), obs_grads[:,None])

        # compute the Jacobian vector product
        grad_input = Jac.matmul(grad_output[:, :, None])

        return grad_input, None

class SIRDatasetPE_obs(torch.utils.data.Dataset):

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
        super(SIRDatasetPE_obs, self).__init__()

        # convert designs and prior samples to PyTorch tensors
        self.X = prior
        # self.X = torch.tensor(prior, dtype=torch.float, device=device, requires_grad=False)

        # simulate data
        sim_sir = SIR_SDE_Observations_Simulator.apply
        self.Y = sim_sir(d, device)

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
    
    def update(self, d, device):
        """
        Simulates new data when d is updated.
        """
        
        # simulate data
        sim_sir = SIR_SDE_Observations_Simulator.apply
        self.Y = sim_sir(d, device)
        
def seed_torch(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# --- HYPER-PARAMETERS --- #

# Parameter dimension; Not relevant for now
modelparams = dict()

# Design Dimensions
modelparams['DIM'] = 3

# Network Params
modelparams['L'] = 4
modelparams['H'] = 20 # [40, 30, 20, 10]

# Sizes
modelparams['data_size'] = 20000
modelparams['num_epochs'] = 50000 # 30000

# Optimisation Params: Psi
modelparams['lr_psi'] = 1e-4
modelparams['step_psi'] = 5000
modelparams['gamma_psi'] = 1 # 0.5 # 0.5 # 1.0

# Optimisation Params: Designs
modelparams['lr_d'] = 3 * 1e-2 # 5 * 1e-3 # 5 * 1e-2 # 0.5 * 1e-1
modelparams['step_d'] = 5000
modelparams['gamma_d'] = 1 # 0.5 # 1.0

# design bounds
bounds = [0, 100]
INIT = 'uniform'

# filename
part = 5
FILENAME = '../data/sde_pe_obs_D{}_traindata_part{}.pt'.format(modelparams['DIM'], part)

# all the flags
Filter = True  # Reject bad / trivial SDE solves
Central = False  # Use centralised gradients
PRINT_D = False  # Print design to tqdm bar
via_NWJ = True  # evaluate JSD lower bound on NWJ lower bound
CLAMP = True
CLAMP_VAL = 2  # heuristic?
SEED = 12345679
# SEED = 12345678
seed_torch(SEED)

# Load initial model and initial designs (check file below)
if part == 1:
    RELOAD = False
else:
    RELOAD = True

# --- DATA PREPARATION --- #

data = torch.load('../data/sir_sde_data.pt')
# data = torch.load('../data/sir_sde_data_large.pt')

if Central:
    grads_central = 0.5 * (data['ys'][2:,:,:] - data['ys'][:-2,:,:]) / (data['ts'][1] - data['ts'][0]) 
    data['grads'] = grads_central
    data['ys'] = data['ys'][1:]  # need to get rid of the first element
else:
    grads_forward = (data['ys'][1:,:,:] - data['ys'][:-1,:,:]) / (data['ts'][1] - data['ts'][0])
    data['grads'] = grads_forward
    
if Filter:
    # find the indices corresponding non-trivial solutions
    idx_good = np.where(np.mean(data['ys'][:,:,1].data.numpy(), axis=0) >= 1)[0]
    data['ys'] = data['ys'][:,idx_good,:]
    data['grads'] = data['grads'][:,idx_good,:]
    data['prior_samples'] = data['prior_samples'][idx_good]

# --- TRAINING PREP --- #

# task specific things
task = 'pe'
loss_function = jsd_loss

# Load Hyper-Parameters if wanted
if RELOAD:
    
    if part >= 10:
        fil_load = FILENAME[:-5] + str(part-1) + '.pt'
    else:
        fil_load = FILENAME[:-4] + str(part-1) + '.pt'
    meta_info = torch.load(fil_load)
    
    # designs
    d_init = meta_info['d_init']
    d = torch.tensor(meta_info['designs_train_jsd'][-1], dtype=torch.float, requires_grad=True, device=device)
    designs = [torch.tensor(dd, dtype=torch.float, device=device) for dd in meta_info['designs_train_jsd']]
    
    # initialise model from previous state
    model_init_state = meta_info['model_init_state']
    model_last_state = meta_info['model_jsd']
    model, _ = initialise_model(
        modelparams, device, task=task, model_init_state=model_last_state)
    
    # data containers
    train_loss = [torch.tensor(tl, dtype=torch.float, device=device) for tl in meta_info['train_loss_jsd']]
    train_loss_viaNWJ = [torch.tensor(tl, dtype=torch.float, device=device) for tl in meta_info['train_loss_jsd_viaNWJ']]

else:
    
    # initialise design
    d, d_init = initialise_design(
        modelparams, device, bounds=bounds, d_init=None, init_type=INIT)
    
    # randomly initialise neural network
    model, model_init_state = initialise_model(
        modelparams, device, task=task, model_init_state=None)
    
    # data containers
    designs = [d.clone().detach()]
    train_loss = list()
    train_loss_viaNWJ = list()

print("Initial Design:", np.sort(d_init.reshape(-1).astype(np.int16)))

# Define Optimizers and Schedulers
optimizer_psi = torch.optim.Adam(model.parameters(), lr=modelparams['lr_psi'], amsgrad=True)
optimizer_design = torch.optim.Adam([d], lr=modelparams['lr_d'], amsgrad=True)
scheduler_psi = StepLR(optimizer_psi, step_size=modelparams['step_psi'], gamma=modelparams['gamma_psi'])
scheduler_design = StepLR(optimizer_design, step_size=modelparams['step_d'], gamma=modelparams['gamma_d'])

if RELOAD:
    
    # load in optimizer state dicts
    optimizer_psi.load_state_dict(meta_info['optimizer_psi_state'])
    optimizer_design.load_state_dict(meta_info['optimizer_design_state'])
    scheduler_psi.load_state_dict(meta_info['scheduler_psi_state'])
    scheduler_design.load_state_dict(meta_info['scheduler_design_state'])
    
    del meta_info

# --- TRAINING --- #

#optimizer_design.param_groups[0]['lr'] = 0.5 * 1e-2
#modelparams['lr_d'] = 0.5 * 1e-2
#optimizer_design.param_groups[0]['lr'] = 0.5 * 1e-4
#modelparams['lr_psi'] = 0.1 * 1e-4
# print(optimizer_psi.param_groups)
#for group in optimizer_psi.param_groups:
#    group['lr'] = 5 * 1e-4
#modelparams['lr_psi'] = 5 * 1e-4

print(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))

# initialize dataset
dataset = SIRDatasetPE_obs(d, data['prior_samples'], device)

# training loop
pbar = tqdm(range(modelparams['num_epochs']), leave=True, disable=False)
for epoch in pbar:
    
    # update samples in dataset
    dataset.update(d, device)

    # get shuffled data
    idx = np.random.choice(
        range(len(data['prior_samples'])), size=modelparams['data_size'], replace=False)
    x, y = dataset[idx]

    # move to device if not there yet
    x, y = x.to(device), y.to(device)

    # compute loss
    loss = loss_function(x, y, model, device)

    # Zero grad the NN optimizer
    optimizer_psi.zero_grad()
    optimizer_design.zero_grad()

    # Back-Propagation
    loss.backward()
    
    if CLAMP:
        with torch.no_grad():
            for param in model.parameters():
                param.grad.clamp_(-CLAMP_VAL, CLAMP_VAL)

    # Perform opt steps for NN
    # optimizer_design.step()
    optimizer_psi.step()
    optimizer_design.step()

    # save a few things to lists
    train_loss.append(-loss.clone().detach())

    if via_NWJ:
        tl = nwj_loss(x, y, lambda a, b: model(a, b) + 1, device)
        train_loss_viaNWJ.append(-tl.clone().detach())

        if PRINT_D:
            pbar.set_postfix(
                MI='{:.3f}'.format(-tl.data.numpy()),
                JSD='{:.3f}'.format(-loss.data.numpy()),
                d=np.sort(d.data.numpy().reshape(-1)).astype(np.int16))
        else:
            pbar.set_postfix(
                MI='{:.3f}'.format(-tl.data.numpy()),
                JSD='{:.3f}'.format(-loss.data.numpy()))
    else:
        if PRINT_D:
            pbar.set_postfix(
                JSD='{:.3f}'.format(-loss.data.numpy()),
                d=np.sort(d.data.numpy().reshape(-1)).astype(np.int16))
        else:
            pbar.set_postfix(
                JSD='{:.3f}'.format(-loss.data.numpy()))

    # LR scheduler step for psi and designs
    scheduler_psi.step()
    scheduler_design.step()

    # Clamp design if beyond boundaries
    with torch.no_grad():
        d.clamp_(bounds[0], bounds[1])

    # Save designs to list
    designs.append(d.clone().detach())
    
# --- SAVE DATA --- #

# clean up lists
train_loss = np.array([mi.cpu().data.numpy() for mi in train_loss])
train_loss_viaNWJ = np.array([mi.cpu().data.numpy() for mi in train_loss_viaNWJ])
designs = np.array([dd.cpu().tolist() for dd in designs])

# create save_dict
save_dict = dict()
save_dict['seed'] = SEED
save_dict['modelparams_jsd'] = modelparams
save_dict['d_init'] = d_init
save_dict['model_init_state'] = model_init_state
save_dict['designs_train_jsd'] = designs
save_dict['model_jsd'] = model.state_dict()
save_dict['train_loss_jsd'] = train_loss
save_dict['train_loss_jsd_viaNWJ'] = train_loss_viaNWJ
save_dict['optimizer_psi_state'] = optimizer_psi.state_dict()
save_dict['optimizer_design_state'] = optimizer_design.state_dict()
save_dict['scheduler_psi_state'] = scheduler_psi.state_dict()
save_dict['scheduler_design_state'] = scheduler_design.state_dict()

# save data
torch.save(save_dict, FILENAME)