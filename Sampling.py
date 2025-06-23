# -*- coding: utf-8 -*-
"""
This script samples the posterior distribution with AW-HMC 
Correction of model misspecification in fracture conductivitites

Related paper : 
When Cubic Law and Darcy Fail: Bayesian Correction of Model Misspecification in Fracture Conductivities

See Perez et al. (2023), Journal of Computational Physics
https://doi.org/10.1016/j.jcp.2023.112342
for full methodological details on the AW-HMC sampler 

See Guiltina et al. (2025). Fractures with variable roughness and wettability (Digital Rocks Portal)
https://www.doi.org/10.17612/p522-cc94 
for the synthetic fracture geometries

"""

import torch
import hamiltorch
import torch.nn as nn
import numpy as np
import util
import scipy.io as sio

import sys
import csv

from scipy.ndimage import gaussian_filter

sys.path.insert(0, "../")
    
print(f'Is CUDA available?: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', torch.cuda.get_device_name(0))

args_list = [123, 50, 200, 100, 1e-3, 10, 0.1, 3, 32, 3]
# rand, burn, Nsamples, L, dt, prior-std, like-std, layers, nn, objective-nbs 

hamiltorch.set_random_seed(int(args_list[0]))
rand = int(args_list[0])
np.random.seed(int(args_list[0]))

# =============================================================================
# Fracture data
# =============================================================================

# Estimation of K_NS for each fracture geometry
mdict_frac = {'f15':0.0195985, 
             'f175':0.0190365, 
             'f20':0.0184925, 
             'f225':0.0174318,
             'f0':0.0209939}

no_rough = False

frac_dim = 225
lu = 1e-4 # Scaling factor for dimensionless inference
voxel_size = 3.94e-6 # Voxel size or resolution of the fracture (dx = 3.94 micrometers)
L_dim = voxel_size/lu # Dimensionless resolution
#ind = 128

X = []; Y = []; Top = []; Bottom = []

with open('./data/Fracture_{}.csv'.format(frac_dim), newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        X.append(row['X'])
        Y.append(row['Y'])
        Top.append(row['Top'])
        Bottom.append(row['Bottom'])
        
x = np.array(X, dtype = np.float64).reshape((256,128)).T*L_dim
y = np.array(Y, dtype = np.float64).reshape((256,128)).T*L_dim
t = np.array(Top, dtype = np.float64).reshape((256,128)).T*L_dim 
b = np.array(Bottom, dtype = np.float64).reshape((256,128)).T*L_dim 

s = 1.5
t = gaussian_filter(t, sigma = s)
b = gaussian_filter(b, sigma = s)

if no_rough == True:
    frac_dim = 0
    t = 0.87*np.ones_like(t)
    b = 0.37*np.ones_like(t)
    
a_m = (t - b) # Mechanical aperture 

Ny = a_m.shape[0]; Nx = a_m.shape[1]
dx = x[0,1] - x[0,0]; dy = y[1,1] - y[0,1] 

# =============================================================================
# Fracture characteristics
# =============================================================================

mean_a_m = np.mean(a_m); std_a_m = np.std(a_m)

if no_rough == True: 
    std_a_m = 1e-12

print('Mean aperture {} μm'.format(int(mean_a_m*lu/1e-6)))
print('Relative roughness {:.2f} (dimensionless)'.format(mean_a_m/std_a_m)) # Ratio of mean to standard deviations of aperture field 
print('Std of aperture {} μm'.format(std_a_m*lu/1e-6))

z2 = np.zeros((2,2))
M = t.shape[0]*t.shape[1]
am_i = np.zeros((t.shape[0]))
am_j = np.zeros((t.shape[1]))

for j in range(t.shape[1]):
    am_j[j] = np.mean(a_m[:,j])
    for i in range(t.shape[0]-1):
        z2[0,0] += ( (t[i+1, j] - t[i, j])/dx )**2  
        z2[1,0] += ( (b[i+1, j] - b[i, j])/dx )**2 
        
for i in range(t.shape[0]):
    am_i[i] = np.mean(a_m[i,:])
    for j in range(t.shape[1]-1):
        z2[0,1] += ( (t[i, j+1] - t[i, j])/dy )**2  
        z2[1,1] += ( (b[i, j+1] - b[i, j])/dy )**2
        
z2 = np.sqrt(z2/M)
JRC = 98.718*z2**1.6833

print('JRC = {:.2f}'.format(np.mean(JRC)))

K_NS = mdict_frac['f{}'.format(frac_dim)]
print('Computed permeability Stokes', K_NS)

# =============================================================================
# Fracture maps
# =============================================================================

K_CL = a_m**2/12 # Local Cubic Law permeability map

dam_dx = np.zeros_like(a_m) ; dam_dy = np.zeros_like(a_m)
dam_dx[:-1, :] = (a_m[1:, :] - a_m[:-1, :])/dx ; dam_dy[:, :-1] = (a_m[:, 1:] - a_m[:, :-1])/dy
dam_dx[-1, :] = dam_dx[-2, :] ; dam_dy[-1, :] = dam_dy[-2, :]

grad_am = (dam_dx**2 + dam_dy**2) # |nabla a_m|²

# =============================================================================
# Neural Network - Training data, Hyperparameters settings & Architecture
# =============================================================================

Ntrain = 10000
x_ = x[:,:].flatten()[:, None]; y_ = y[:,:].flatten()[:, None]

idx = np.random.choice(len(x_), Ntrain, replace = False)
x_train = x_[idx,:]; y_train = y_[idx,:]

XY_train = torch.tensor( np.concatenate([x_train, y_train], 1), dtype = torch.float32, device = device )
a_m_train  = torch.tensor ( a_m.flatten()[:, None][idx, :], dtype = torch.float32, device = device )
grad_am_train  = torch.tensor ( grad_am.flatten()[:, None][idx, :], dtype = torch.float32, device = device )

data = {}
data['XY'] = torch.reshape( XY_train, (Ntrain, 2))
data['a_m'] = torch.reshape( a_m_train, (Ntrain, 1))
data['grad_am'] = torch.reshape( grad_am_train, (Ntrain, 1))

for d in data:
    data[d] = data[d].to(device)
    
del XY_train, a_m_train, grad_am_train

#################### Neural Network hyperparameters - AW-HMC sampler

obj_nb_ll = int(args_list[9]); # Number of objectives in the multi-objective problem (excluding the priors)
n_params_single = 1 # Number of inverse parameters (only the correction factor alpha here)

activation = torch.sin 
pde = True

N = int(args_list[1]) # Number of adaptive steps for AW-HMC
num_samples = int(args_list[2]) # Number of samples for the inference with AW-HMC

L = int(args_list[3]) # Leapfrog number of iterations for AW-HMC
step_size = float(args_list[4]) # Leapfrog time step for AW-HMC

prior_std = float(args_list[5]) # Prior standard deviation
like_std = float(args_list[6]) # Likelihood standard deviation  

tau_priors = 1/prior_std**2 # Lambda_{k+1} for the prior term
tau_likes = [1/like_std**2]*(obj_nb_ll) # List of lamba_k for the objectives in model_loss function

ll = int(args_list[7]); num_nodes = int(args_list[8]); 
layers = [2] + ll*[num_nodes] + [1]

N_params = 0
for i in range(len(layers)-2):
    N_params += (layers[i]*layers[i+1])+layers[i+1] 
N_params += (layers[-2]*layers[-1])+layers[-1]

print("Number of network parameters", N_params, "Inverse parameters", n_params_single)

####################### Neural Network architecture 

class Layer(nn.Module):

    def __init__(self, in_features, out_features):
        super(Layer, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        return self.linear(x)

class Sin_red(nn.Module):
    
    def __init__(self, activation):
        super(Sin_red, self).__init__()
        self.activation = activation
        
    def forward(self, x):
        return 0.5*(1.0+self.activation(x)) 

class Activation(nn.Module):
    
    def __init__(self, activation):
        super(Activation, self).__init__()
        self.activation = activation 
        
    def forward(self, x):
        return self.activation(x)

class Net_frac(nn.Module):

    def __init__(self, sizes, activation):
        super(Net_frac, self).__init__()

        layer = []
        for i in range(len(sizes)-2):
            linear = Layer(sizes[i], sizes[i+1])
            act_func = Activation(activation)
            layer += [linear, act_func]

        layer += [Layer(sizes[-2], sizes[-1]), Sin_red(activation)]

        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)
    
net_frac = Net_frac(layers, activation).to(device)
nets = [net_frac]

# =============================================================================
# Models
# =============================================================================

def model_loss_sampling(data, fmodel, params_unflattened, tau_likes, gradients, params_single = None):
    xy = data['XY'] 
    Im_a = data['a_m'][:,0]    
    Im_grad_a = data['grad_am'][:,0]
    
    xy = xy.detach().requires_grad_()
        
    pred_a = fmodel[0](xy, params=params_unflattened[0])[:,0]
    da = gradients(pred_a, xy)[0] 
    pred_grad_a = torch.sum(da**2, dim = 1)

    pred_r = torch.abs(pred_a - mean_a_m)/std_a_m
    pred_K = pred_a**2/12
    
    pred_a = torch.minimum(pred_a, Im_a)
    alpha = torch.exp(params_single[0])

    ll1 = - 0.5 * tau_likes[0] * ( (pred_a - Im_a) ** 2).sum(0)
    ll2 = - 0.5 * tau_likes[1] * ( (pred_grad_a - Im_grad_a) ** 2).sum(0)
    ll3 = - 0.5 * tau_likes[2] * ( (pred_K*(1+alpha*pred_r)).mean(0) - K_NS ) ** 2 * Ntrain 
 
    ll_list = [ll1, ll2, ll3]
    output = [pred_a]

    if torch.cuda.is_available():
        del xy, Im_a, Im_grad_a, pred_a, da, pred_grad_a, pred_r, pred_K
        torch.cuda.empty_cache()

    return ll_list, output

# =============================================================================
# Sampling 
# =============================================================================

sampler = hamiltorch.Sampler.AW_HMC
params_hmc, res_p, res_m, ham_table, grad_list, l_list, AR, lambda_list = util.sample_model_bpinns(nets, data, model_loss = model_loss_sampling, n_params_single = n_params_single, debug = 1, num_samples = num_samples, num_steps_per_sample = L, step_size = step_size, num_adap_step = N, tau_priors = tau_priors, tau_likes = tau_likes, device = device, pde = pde, sampler = sampler)

samples = torch.stack(params_hmc).cpu().detach().numpy()
post = torch.stack(l_list).cpu().detach().numpy()

res_params = torch.stack(res_p).cpu().detach().numpy()
res_momentum = torch.stack(res_m).cpu().detach().numpy()

# Sampling parameters 
mdict = {
"samples": samples,
"post": post, 
"momentum" : res_momentum, 
"params" : res_params, 
"ham_table" : ham_table, 
"AR":AR, 
"lambda_list": lambda_list
}

name = 'BPINN_fracture_f{}_ll{}_num_nodes{}_stdll{}_stdp{}_L{}_dt{}_ntrain{}_Nsamples{}_burn{}_rand{}'.format(frac_dim, ll, num_nodes, like_std, prior_std,  L, step_size, Ntrain, num_samples, N, rand)
sio.savemat("./results_sampling/"+name, mdict)
