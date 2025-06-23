# -*- coding: utf-8 -*-
"""
This script performs the prediction of the Bayesian posterior distribution 
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
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import util
import scipy.io as sio

import sys
import csv

from scipy.ndimage import gaussian_filter
from matplotlib.gridspec import GridSpec 

sys.path.insert(0, "../")

print(f'Is CUDA available?: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', torch.cuda.get_device_name(0))

args_list = [123, 50, 200, 100, 1e-3, 10, 0.1, 3, 32, 3]
# rand, burn, Nsamples, L, dt, prior-std, like-std, layers, nn, objective-nbs

filename = "./results_sampling/"
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
# Neural Network - Validation data, Hyperparameters settings & Architecture
# =============================================================================

N_val = Nx*Ny; Ntrain = 10000
x_ = x[:,:].flatten()[:, None]; y_ = y[:,:].flatten()[:, None]

XY = torch.tensor( np.concatenate([x_, y_], 1), dtype = torch.float32, device = device ) 

data_val = {}
data_val['XY'] = torch.reshape( XY, (Nx*Ny, 2) )
data_val['a_m'] = torch.tensor( a_m.reshape(Nx*Ny, 1), dtype=torch.float32, device=device ) 
data_val['grad_am'] = torch.tensor( grad_am.reshape(Nx*Ny, 1), dtype=torch.float32, device=device ) 

for d in data_val:
    data_val[d] = data_val[d].to(device)

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

def model_loss_pred(data, fmodel, params_unflattened, tau_likes, gradients, params_single = None):
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
    ll3 = - 0.5 * tau_likes[2] * ( (pred_K*(1+alpha*pred_r)).mean(0) - K_NS ) ** 2 * N_val
 
    ll_list = [ll1, ll2, ll3]
    output = [pred_a]

    if torch.cuda.is_available():
        del xy, Im_a, Im_grad_a, pred_a, da, pred_grad_a, pred_r, pred_K
        torch.cuda.empty_cache()

    return ll_list, output



# =============================================================================
# Prediction
# =============================================================================
name = 'BPINN_fracture_f{}_ll{}_num_nodes{}_stdll{}_stdp{}_L{}_dt{}_ntrain{}_Nsamples{}_burn{}_rand{}'.format(frac_dim, ll, num_nodes, like_std, prior_std,  L, step_size, Ntrain, num_samples, N, rand)
data = sio.loadmat(filename+name)

sampler = hamiltorch.Sampler.AW_HMC
# params_hmc = torch.tensor(data['samples'], dtype = torch.float32, device = device); burn = 0 
params_hmc = torch.tensor(data['params'][:, -1, :], dtype = torch.float32, device = device); burn = N

lambda_list = data['lambda_list']
ham_table = data['ham_table']

pred_list = util.predict_model_bpinns(nets, params_hmc, data_val, n_params_single = n_params_single, model_loss = model_loss_pred, tau_priors = tau_priors, tau_likes = tau_likes, lambda_list = list(lambda_list[-1,:-1]), pde = pde)
pred_list_a = pred_list[0].cpu().numpy() 

pred_mean_a = pred_list_a[burn:,:].mean(0).reshape(Ny, Nx)
pred_std_a = pred_list_a[burn:,:].std(0).reshape(Ny, Nx)

# =============================================================================
# Results & Analysis 
# =============================================================================

micro_dim = lu/1e-6 # Recover the dimensions for the aperture fields and express them in micrometers

fig = plt.figure(figsize=(18,9), constrained_layout=True)
s = 0.8; p = 0.05
nbins = 30

gs = GridSpec(1, 2, figure = fig, width_ratios=[2, 1])
gs0 = gs[0].subgridspec(2, 2)
gs1 = gs[1].subgridspec(2, 1, height_ratios=[1.4, 1])

ax = fig.add_subplot(gs1[0,0], projection='3d')
ax.plot_surface(x*micro_dim, y*micro_dim, t*micro_dim, cmap = 'binary')
ax.plot_surface(x*micro_dim, y*micro_dim, b*micro_dim, cmap = 'binary') 
ax.view_init(elev = 20, azim = -138)
ax.set_box_aspect([6,3,1])
ax.tick_params(axis='both', labelsize = 9)
ax.set_xlabel('x') ; ax.set_ylabel('y');  ax.set_zlabel('z'); 
ax.xaxis.labelpad = 20 ; ax.yaxis.labelpad = 10 ; ax.zaxis.labelpad = 2 
ax.grid(False)

ax = fig.add_subplot(gs0[0,0])
im = ax.imshow(a_m*micro_dim, extent = [0, x[-1,-1]*micro_dim, y[-1,-1]*micro_dim, 0])
ax.set_title('Original data on \n'+r'mechanical aperture map $a_m$ ($\mu m$)')
ax.set_xlabel(r'$x$ ($\mu m$)')
ax.set_ylabel(r'$y$ ($\mu m$)')
fig.colorbar(im, ax = ax, orientation = 'horizontal', shrink = s, pad = p)

ax = fig.add_subplot(gs0[0,1])
im = ax.imshow(pred_mean_a*micro_dim, extent = [0, x[-1,-1]*micro_dim, y[-1,-1]*micro_dim, 0])
ax.set_title('Mean prediction on \n'+r'hydraulic aperture map $a_h$ ($\mu m$)')
ax.set_xlabel(r'$x$ ($\mu m$)')
ax.set_ylabel(r'$y$ ($\mu m$)')
fig.colorbar(im, ax = ax, orientation = 'horizontal', shrink = s, pad = p)

ax = fig.add_subplot(gs0[1,0])
im = ax.imshow(np.abs(pred_mean_a*micro_dim - a_m*micro_dim)/(a_m*micro_dim), extent = [0, x[-1,-1]*micro_dim, y[-1,-1]*micro_dim, 0])
ax.set_title('Absolute relative error \n'+r'between $a_h$ and $a_m$')
ax.set_xlabel(r'$x$ ($\mu m$)')
ax.set_ylabel(r'$y$ ($\mu m$)')
fig.colorbar(im, ax = ax, orientation = 'horizontal', shrink = s, pad = p)

ax = fig.add_subplot(gs0[1,1])
im = ax.imshow(pred_std_a*micro_dim, extent = [0, x[-1,-1]*micro_dim, y[-1,-1]*micro_dim, 0])
ax.set_title('Uncertainty on \n'+r'hydraulic aperture map $a_h$ ($\mu m$)')
ax.set_xlabel(r'$x$ ($\mu m$)')
ax.set_ylabel(r'$y$ ($\mu m$)')
fig.colorbar(im, ax = ax, orientation = 'horizontal', shrink = s, pad = p)

ax = fig.add_subplot(gs1[1,0])
ax.hist((a_m*micro_dim).flatten(), bins = nbins, histtype = 'stepfilled', alpha = 0.3, label = r'Initial distribution on $a_m$')
ax.axvline(np.mean(a_m*micro_dim), linestyle = '--', color = 'black', alpha = 0.2, label = r'$<a_m>$')
ax.axvline(np.mean(a_m*micro_dim)+np.std(a_m*micro_dim), linestyle = '-', color = 'black', alpha = 0.2, label = r'$<a_m>\pm\sigma_{a_m}$')
ax.axvline(np.mean(a_m*micro_dim)-np.std(a_m*micro_dim), linestyle = '-', color = 'black', alpha = 0.2)
ax.hist((pred_mean_a*micro_dim).flatten(), bins = nbins, histtype = 'stepfilled', alpha = 0.3, color = 'red', label = r'Final distribution on $a_h$')
ax.set_xlabel(r'Aperture values ($\mu m$)')
ax.legend()
ax.set_ylim(ymax = 3600)
ax.xaxis.set_tick_params(labelbottom=True)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_yticks([])

plt.savefig('Hydraulic_aperture.pdf')


#%%
plt.figure(figsize = (5.5,7.5), constrained_layout=True)
plt.subplot(2,1,1); 
pred_alpha = np.exp(data['params'][:,:,0].flatten())
plt.scatter(pred_alpha[:N*L], data['momentum'][:,:,0].flatten()[:N*L], color = 'red', alpha = 0.5, s = 4, label = 'Adaptive steps trajectory')
plt.scatter(pred_alpha[N*L:], data['momentum'][:,:,0].flatten()[N*L:], alpha = 0.5, s = 4, label = r'Trajectory $\tau>N$')
plt.legend(loc = 'lower left')
plt.xlabel(r'Parameter $\alpha$')
plt.ylabel('Momemtum')
plt.grid(alpha = 0.3)

plt.subplot(2,1,2); 
plt.hist(pred_alpha[N*L:], bins = 10,histtype = 'stepfilled', alpha = 0.3, label = ' Posterior distribution\n' + r'after adaptation $\tau>N$')
plt.hist(pred_alpha[:N*L], bins = 40, histtype = 'stepfilled', alpha = 0.5, color = 'red', label = '   Adaptive steps\n' + r'convergence $\tau\leq N$')
plt.axvline(x=np.mean(pred_alpha[N*L:]), linestyle = '--', color = 'black', alpha = 0.3, label = '  Posterior mean\n' + r'estimate $\bar\alpha = 0.038$')
ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=True)
ax.yaxis.set_tick_params(labelleft=False)
plt.legend(loc = 'upper right')
plt.xlabel(r'Parameter $\alpha$')
ax.set_yticks([])

plt.savefig('Inv_Param.pdf')


#%%
K_dim = lu**2/1e-12
KCL = a_m**2/12

pred_K = (pred_list_a**2/12)*(1+np.mean(pred_alpha[N*L:])*np.abs(pred_list_a-mean_a_m)/std_a_m) 
pred_K_mean = pred_K[burn:,:].mean(0).reshape(Ny, Nx)
pred_K_std = pred_K[burn:,:].std(0).reshape(Ny, Nx)

print(r'Corrected upscaled permeability value (mean): {:.1f} ($\mu m^2$)'.format(np.mean(pred_K_mean)*K_dim))
print('Corrected upscaled permeability value (range): [{:.1f}, {:.1f}] ($\mu m^2$)'.format(np.mean(pred_K_mean-pred_K_std)*K_dim, np.mean(pred_K_mean+pred_K_std)*K_dim))
print('Stokes permeability', K_NS)

fig = plt.figure(figsize=(18,9), constrained_layout=True)
s = 0.8; p = 0.05

gs = GridSpec(1, 2, figure = fig, width_ratios=[2, 1])
gs0 = gs[0].subgridspec(2, 2)
gs1 = gs[1].subgridspec(2, 1, height_ratios=[0.8, 0.8])

ax = fig.add_subplot(gs0[0,0])
im = ax.imshow(KCL*K_dim, extent = [0, x[-1,-1]*micro_dim, y[-1,-1]*micro_dim, 0])
ax.set_title('Original permeability\n'+r' map $K_{CL}^{ a_m}(x,y)$')
ax.set_xlabel(r'$x$ ($\mu m$)')
ax.set_ylabel(r'$y$ ($\mu m$)')
fig.colorbar(im, ax = ax, orientation = 'horizontal', shrink = s, pad = p)

ax = fig.add_subplot(gs0[0,1])
im = ax.imshow(pred_K_mean*K_dim, extent = [0, x[-1,-1]*micro_dim, y[-1,-1]*micro_dim, 0], vmin = np.min(KCL*K_dim), vmax = np.max(KCL*K_dim))
ax.set_title('Mean prediction on \n'+r'permeability map $K_{NN}^{ a_h}(x,y)$')
ax.set_xlabel(r'$x$ ($\mu m$)')
ax.set_ylabel(r'$y$ ($\mu m$)')
fig.colorbar(im, ax = ax, orientation = 'horizontal', shrink = s, pad = p)


ax = fig.add_subplot(gs0[1,0])
im = ax.imshow(np.abs(pred_K_mean*K_dim-KCL*K_dim)/(KCL*K_dim), extent = [0, x[-1,-1]*micro_dim, y[-1,-1]*micro_dim, 0])
ax.set_title('Absolute relative error \n'+r'between $K_{CL}^{ a_m}$ and $K_{NN}^{ a_h}$')
ax.set_xlabel(r'$x$ ($\mu m$)')
ax.set_ylabel(r'$y$ ($\mu m$)')
fig.colorbar(im, ax = ax, orientation = 'horizontal', shrink = s, pad = p)

ax = fig.add_subplot(gs0[1,1])
im = ax.imshow(pred_K_std*K_dim, extent = [0, x[-1,-1]*micro_dim, y[-1,-1]*micro_dim, 0])
ax.set_title('Uncertainty on \n'+r'permeability map $K_{NN}^{ a_h}(x,y)$')
ax.set_xlabel(r'$x$ ($\mu m$)')
ax.set_ylabel(r'$y$ ($\mu m$)')
fig.colorbar(im, ax = ax, orientation = 'horizontal', shrink = s, pad = p)

ax = fig.add_subplot(gs1[1,0])
ax.hist((K_CL*K_dim).flatten(), bins = nbins,histtype = 'stepfilled', alpha = 0.3, label = r'Initial distribution on $K_{CL}^{a_m}$')
ax.hist((pred_K_mean*K_dim).flatten(), histtype = 'stepfilled', bins = nbins, alpha = 0.2, color = 'green', label = r'Final distribution on $K_{NN}^{a_h}$')
ax.axvline(np.mean(a_m)**2/12*K_dim, linestyle = '--', color = 'red', alpha = 0.2, label = r'$K_{CL}$ = '+'{:.2f}'.format(208.33))
ax.axvline(K_NS*K_dim, linestyle = '--', color = 'green', alpha = 0.2, label = r'$K_{NS}$ = '+'{:.2f}'.format(174.31))
ax.legend()
ax.set_xlabel(r'Permeability values ($\mu m^2$)')
ax.set_ylim(ymax = 3200)
ax.xaxis.set_tick_params(labelbottom=True)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_yticks([])

burn = 0
err = np.zeros((num_samples,))
K_pred = np.zeros((num_samples,))
K_pred_std_plus = np.zeros((num_samples,))
K_pred_std_moins = np.zeros((num_samples,))

pred_K = (pred_list_a**2/12)*(1+np.mean(pred_alpha[N*L:])*np.abs(pred_list_a-mean_a_m)/std_a_m) 

for i in range(num_samples-burn):
    pred_a = pred_list_a[burn+i:burn+i+1,:].mean(0).reshape(Ny, Nx)
    err[i] += np.sum( (pred_a - a_m)**2 )/(Nx*Ny)
    pred_K_mean = pred_K[burn+i:burn+i+1,:].mean(0).reshape(Ny, Nx)
    pred_K_std = pred_K[N:N+i+1,:].std(0).reshape(Ny, Nx)
    K_pred[burn+i] = np.mean(pred_K_mean)
    K_pred_std_plus[burn+i] = np.mean(pred_K_mean+pred_K_std)
    K_pred_std_moins[burn+i] = np.mean(pred_K_mean-pred_K_std)
    

from math import nan
K_pred[:burn] = nan
K_pred_std_plus[:N] = nan
K_pred_std_moins[:N] = nan

ax = fig.add_subplot(gs1[0,0])
ax.plot(K_pred*K_dim, label = r'Upscaled mean on $K_{NN}^{ a_h}$')
ax.fill_between(np.arange(0,20), K_pred_std_moins*K_dim, K_pred_std_plus*K_dim, alpha = 0.2, label = r'Cumulated uncertainty on $K_{NN}^{ a_h}$')
ax.axhline(K_NS*K_dim, linestyle = '--', color = 'green', alpha = 0.5, label = r'$K_{NS} = 174.31$')
ax.axhline(np.mean(a_m)**2/12*K_dim, linestyle = '--', color = 'red', alpha = 0.5, label = r'$K_{CL} = 208.33$')
ax.axvline(N, linestyle = '--', color = 'grey', alpha = 0.5)
ax.set_ylim(ymin = 4e-3*K_dim)

ax.annotate('', 
           xy = (0, 5e-3*K_dim), xycoords = 'data', 
           xytext = (45, 5e-3*K_dim), textcoords = 'data', 
           arrowprops=dict(arrowstyle='<->', color = 'grey'))
ax.annotate('Adaptive steps', 
           xy = (0, 4.5e-3*K_dim), xycoords = 'data',
           xytext = (0, 4.5e-3*K_dim), textcoords = 'data', 
           color = 'grey', size = 9)
ax.legend(loc = 'lower right')
ax.set_xlabel('Sampling steps\n'+'')
ax.set_ylabel(r'Permeability estimates ($\mu m^2$)')

plt.savefig('Permeability.pdf')

#%%

mdict = {
"pred_K": pred_K_mean, 
"pred_K_std":pred_K_std, 
"pred_ah":pred_mean_a, 
"pred_ah_std":pred_std_a
}

name = 'K_ah_pred_f{}.mat'.format(frac_dim)
sio.savemat("./results_prediction/"+name, mdict)
