import torch
from enum import Enum
import numpy as np

from . import util
from math import sqrt 

class Sampler(Enum):
    HMC = 1
    AW_HMC = 2
    HMC_NUTS = 3
    
def collect_gradients(log_prob, params):
    """Returns the parameters and the corresponding gradients (params.grad).

    Parameters
    ----------
    log_prob : torch.tensor
        Tensor shape (1,) which is a function of params (Can also be a tuple where log_prob[0] is the value to be differentiated).
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters .

    Returns
    -------
    torch.tensor
        The params, which is returned has the gradient attribute attached, i.e. params.grad.

    """

    if isinstance(log_prob, tuple):
        log_prob[0].backward()
        params_list = list(log_prob[1])
        params = torch.cat([p.flatten() for p in params_list])
        params.grad = torch.cat([p.grad.flatten() for p in params_list])
    else:
        params.grad = torch.autograd.grad(log_prob, params)[0]
    return params

def gibbs(params, sampler = Sampler.AW_HMC, log_prob_func=None, mass=None):
    """Performs the momentum resampling component of HMC.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.AW_HMC, Sampler.HMC}.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
    mass : torch.tensor or list
        The mass matrix is related to the inverse covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the mass matrix as a list of blocks. Hope to make that more efficient.
   
    Returns
    -------
    torch.tensor
        Returns the resampled momentum vector of shape (D,).

    """
    if mass is None:
        dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params)) 
    else:
        if type(mass) is list:
            # block wise mass list of blocks
            samples = torch.zeros_like(params)
            i = 0
            for block in mass:
                it = block[0].shape[0]
                dist = torch.distributions.MultivariateNormal(torch.zeros_like(block[0]), block)
                samples[i:it+i] = dist.sample()
                i += it
            return samples
        elif len(mass.shape) == 2:
            dist = torch.distributions.MultivariateNormal(torch.zeros_like(params), mass)  
        elif len(mass.shape) == 1:
            dist = torch.distributions.Normal(torch.zeros_like(params), mass ** 0.5) 
    return dist.sample()


def leapfrog(params, momentum, log_prob_func, obj_nb, lambda_, steps=10, step_size=0.1, inv_mass=None, sampler=Sampler.AW_HMC, store_on_GPU = True, debug=False):
    """ Leapfrog integration scheme used for HMC and AW_HMC
    
    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters.
    momentum : torch.tensor
        Flat vector of momentum, corresponding to the parameters: shape (D,), where D is the dimensionality of the parameters.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
    obj_nb : int
        Number of objective terms in the potential energy (including the priors)
    lambda_ : list
        List of the weights for each objective term in the potential energy (including the priors).        
    steps : int
        The number of steps to take per trajector (often referred to as L).
    step_size : float
        Size of each step to take when doing the numerical integration.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D).
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.AW_HMC, Sampler.HMC}.
    store_on_GPU : bool
        Option that determines whether to keep samples in GPU memory. It runs fast when set to TRUE but may run out of memory unless set to FALSE.
    debug : int
        Set to zero for no print statements.

    Returns
    -------
    ret_params : list
        List of parameters collected in the trajectory.
    ret_momenta : list
        List of momentum collected in the trajectory. 
    grad_ll : numpy array 
        Gradients' table of the objective terms. Shape (NO, D, L) where NO is the number of objective terms in the whole potential energy (including the priors),
        D is the dimensionality of the parameters and L the number of leapfrog steps.
    """
    params = params.clone(); momentum = momentum.clone()
    
    if sampler == Sampler.AW_HMC or sampler == Sampler.HMC:
        def params_grad(p):
            p = p.detach().requires_grad_()
            log_prob = []
            grad = []
            for i in range(obj_nb):
                log_prob.append(lambda_[i]*log_prob_func(p)[i]) # List of the weighted objective terms (lambda_ = 1 in Sampler.HMC)
                grad.append(collect_gradients(log_prob[i], p).grad.detach()) # List of gradient tensors (of size D) for each objective term
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return grad 
       
        ret_params = []
        ret_momenta = []

        grad_ll = np.zeros((obj_nb, params.shape[0], steps)) 
        p_grad_all = params_grad(params) # List of tensors of size D for all the objective terms

        p_grad = 0
        for i in range (obj_nb):  
            p_grad += p_grad_all[i] # Gradient of the weighted potential
        momentum += 0.5 * step_size * p_grad

        for n in range(steps):
                    
            if inv_mass is None: # M = Id and inv(M) = Id
                params = params + step_size * momentum 
            else:
                # Assum G is diag here so 1/Mass = G inverse
                if type(inv_mass) is list:
                    i = 0
                    for block in inv_mass:
                        it = block[0].shape[0]
                        params[i:it+i] = params[i:it+i] + step_size * torch.matmul(block,momentum[i:it+i].view(-1,1)).view(-1) 
                        i += it
                elif len(inv_mass.shape) == 2:
                    params = params + step_size * torch.matmul(inv_mass,momentum.view(-1,1)).view(-1)
                else:
                    params = params + step_size * inv_mass * momentum
                    
            p_grad_all = params_grad(params) # Update the gradient of the weighted potential
            p_grad = 0
            for i in range (obj_nb): 
                p_grad += p_grad_all[i]
             
            momentum += step_size * p_grad
                     
            ret_params.append(params.clone())
            ret_momenta.append(momentum.clone())
            
            # Store the gradients for a posteriori diagnostic 
            for i in range(obj_nb):
                if p_grad.is_cuda:
                    grad_ll[i, :, n] = p_grad_all[i].clone().detach().cpu()
                else : 
                    grad_ll[i, :, n] = p_grad_all[i].clone()

        # only need last for Hamiltoninian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
        ret_momenta[-1] = ret_momenta[-1] - 0.5 * step_size * p_grad.clone()
        return ret_params, ret_momenta, grad_ll     
    else:
        raise NotImplementedError()


def acceptance(h_old, h_new):
    """Returns the log acceptance ratio for the Metroplis-Hastings step.

    Parameters
    ----------
    h_old : torch.tensor
        Previous value of Hamiltonian (1,).
    h_new : type
        New value of Hamiltonian (1,).

    Returns
    -------
    float
        Log acceptance ratio.

    """

    return float(-h_new + h_old)


def adaptation(rho, t, step_size_init, H_t, eps_bar, desired_accept_rate=0.8):
    """No-U-Turn sampler adaptation of the step size. This follows Algo 5, p. 15 from Hoffman and Gelman 2011.

    Parameters
    ----------
    rho : float
        rho is current acceptance ratio.
    t : int
        Iteration.
    step_size_init : float
        Initial step size.
    H_t : float
        Current rolling H_t.
    eps_bar : type
        Current rolling step size update.
    desired_accept_rate : float
        The step size is adapted with the objective of a desired acceptance rate.

    Returns
    -------
    step_size : float
        Current step size to be used.
    eps_bar : float
        Current rolling step size update. Also at last iteration this is the final adapted step size.
    H_t : float
        Current rolling H_t to be passed at next iteration.

    """
    # rho is current acceptance ratio
    # t is current iteration
    t = t + 1
    if util.has_nan_or_inf(torch.tensor([rho])):
        alpha = 0 # Acceptance rate is zero if nan.
    else:
        alpha = min(1.,float(torch.exp(torch.FloatTensor([rho]))))
    mu = float(torch.log(10*torch.FloatTensor([step_size_init])))
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    H_t = (1-(1/(t+t0)))*H_t + (1/(t+t0))*(desired_accept_rate - alpha)
    x_new = mu - (t**0.5)/gamma * H_t
    step_size = float(torch.exp(torch.FloatTensor([x_new])))
    x_new_bar = t**-kappa * x_new +  (1 - t**-kappa) * torch.log(torch.FloatTensor([eps_bar]))
    eps_bar = float(torch.exp(x_new_bar))

    return step_size, eps_bar, H_t


def hamiltonian(params, momentum, log_prob_func, obj_nb, lambda_, inv_mass=None, sampler=Sampler.AW_HMC):
    """Computes the Hamiltonian as a function of the parameters and the momentum.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters.
    momentum : torch.tensor
        Flat vector of momentum, corresponding to the parameters: shape (D,), where D is the dimensionality of the parameters.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
    obj_nb : int
        Number of objective terms in the potential energy (including the priors).     
    lambda_ : list
        List of the weights for each objective term in the potential energy (including the priors).
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.AW_HMC, Sampler.HMC}.
    
    Returns
    -------
    torch.tensor
        Returns the value of the Hamiltonian: shape (1,).

    """
    if sampler == Sampler.AW_HMC or sampler == Sampler.HMC:
        log_prob = 0
        for i in range(obj_nb):
            log_prob += lambda_[i]*log_prob_func(params)[i] # Weighted hamiltonian for Inv-Dir
           
        if util.has_nan_or_inf(log_prob):
            print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
            raise util.LogProbError()

        potential = - log_prob  
        if inv_mass is None:
            kinetic = 0.5 * torch.dot(momentum, momentum)  
        else:
            if type(inv_mass) is list:
                i = 0
                kinetic = 0
                for block in inv_mass:
                    it = block[0].shape[0]
                    kinetic = kinetic +  0.5 * torch.matmul(momentum[i:it+i].view(1,-1),torch.matmul(block,momentum[i:it+i].view(-1,1))).view(-1)
                    i += it
            # Assum G is diag here so 1/Mass = G inverse
            elif len(inv_mass.shape) == 2:
                kinetic = 0.5 * torch.matmul(momentum.view(1,-1),torch.matmul(inv_mass,momentum.view(-1,1))).view(-1)
            else:
                kinetic = 0.5 * torch.dot(momentum, inv_mass * momentum)
        hamiltonian = potential + kinetic 
    
    else:
        raise NotImplementedError()
    return hamiltonian


def sample(log_prob_func, params_init, lambda_, num_samples=10, num_steps_per_sample=10, step_size=0.1, burn=-1,  num_adap_step = 20, n_params_single = None, inv_mass=None, sampler=Sampler.AW_HMC, debug=False, store_on_GPU = True):
    """ Sampling function of hamiltorch. This function receives a function handle log_prob_func, which the sampler will use to evaluate the log probability of each sample. 
     A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.

    Parameters
    ----------
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
    params_init : torch.tensor
        Initialisation of the parameters.
    lambda_ : list
        List of the weights for each objective term in the potential energy (including the priors).
    num_samples : int
        Sets the number of samples corresponding to the number of momentum resampling steps/the number of trajectories to sample.
    num_steps_per_sample : int
        The number of steps to take per trajector (often referred to as L).
    step_size : float
        Size of each step to take when doing the numerical integration.
    burn : int
        Number of samples to burn before collecting samples. Set to -1 for no burning of samples. This must be less than `num_samples` as `num_samples` subsumes `burn`.
    num_adap_step: int
        Number of adaptive iterations, referred as N in the Adaptive-Weighted HMC algorithm. 
    n_params_single : int
        The number of single parameters that have to be inferred.        
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.    
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.AW_HMC, Sampler.HMC}.
    debug : {0, 1}
        Debug mode can take 2 options. Setting debug = 0 (default) allows the sampler to run as normal. Setting debug = 1 prints both the old and new Hamiltonians per iteration.
    store_on_GPU : bool
        Option that determines whether to keep samples in GPU memory. It runs fast when set to TRUE but may run out of memory unless set to FALSE.

    Returns
    -------
    param_samples : list of torch.tensor(s)
        A list of parameter samples: end of the leapfrog trajectories.
    leap_params : list of torch.tensor(s)
        Full trajectory of the sampled parameters. Shape (S, L, D) with S the number of samples.
    leap_momentum : list of torch.tensor(s) 
        Full trajectory of the sampled momentum. Shape (S, L, D) with S the number of samples.
    ham_list : numpy array 
        Hamiltonian values, array of shape (S, L) with S the number of samples.
    grad_list : numpy array 
        List of the (weighted in AW-HMC) gradients for a posteriori diagnostic. Shape (NO, D, L, S) with S the number of samples and NO the number of objective terms.
    log_prob_list : list of torch.tensor
        List of log probability values for each sample, shape the number of samples (excluding the burning steps).
    AR : list
        List of rejected samples.
    lambda_list : numpy array
        List of the weights for each objective term in the potential energy (including the priors) and for each adaptive iteration in the AW_HMC algorithm.
    """

    device = params_init.device
    if params_init.dim() != 1:
        raise RuntimeError('params_init must be a 1d tensor.')

    if burn >= num_samples or num_adap_step >= num_samples :
        raise RuntimeError('burn and num_adap_step must be less than num_samples.')

    if burn == -1:
        burn = num_adap_step # Default value for the burning 
        print('Burning will be set to the number of adaptive iterations:', burn)

    NUTS = False
    if sampler == Sampler.HMC_NUTS:
        if burn == 0:
            raise RuntimeError('burn must be greater than 0 for NUTS.')
        sampler = Sampler.HMC
        NUTS = True
        step_size_init = step_size
        H_t = 0.
        eps_bar = 1.

    # Invert mass matrix once (As mass is used in Gibbs resampling step)
    mass = None
    if inv_mass is not None:
        if type(inv_mass) is list:
            mass = []
            for block in inv_mass:
                mass.append(torch.inverse(block))
        # Assum G is diag here so 1/Mass = G inverse
        elif len(inv_mass.shape) == 2:
            mass = torch.inverse(inv_mass)
        elif len(inv_mass.shape) == 1:
            mass = 1/inv_mass
            print(mass[0])

    params = params_init.clone().requires_grad_()
    print('Sampler', params.shape)
   
    if not store_on_GPU:
        ret_params = [params.clone().detach().cpu()]
    else:
        ret_params = [params.clone()]
        
    # ===============================================  AW_HMC ALGORITHM  ===============================================

    obj_nb = len(log_prob_func(params)) # number of objectives in the loss function (including the priors)
    num_rejected = 0
    
    leap_params = [] 
    leap_momentum = []

    lambda_list = np.zeros((num_adap_step+1, obj_nb)) 
    grad_list = np.zeros((obj_nb, params.shape[0], num_steps_per_sample, num_samples))
    ham_list = np.zeros((num_samples, num_steps_per_sample))

    log_prob_list = [] 
    AR = []  
    
    util.progress_bar_init('Sampling {}'.format(sampler), num_samples, 'Samples')
    for n in range(num_samples):
        if n == num_samples - 1:
            print('Rejected samples = ', num_rejected)
        
        util.progress_bar_update(n)
        try:
            momentum = gibbs(params, sampler=sampler, log_prob_func=log_prob_func, mass=mass)
                     
            # ======================= Inverse-Dirichlet weighting strategy  =========================
            if sampler == Sampler.AW_HMC: 
                
                p = params.detach().requires_grad_()
                if n <= num_adap_step : 
                    grad = []; temp = [];
                    for i in range(obj_nb-1):
                        grad.append(collect_gradients(log_prob_func(p)[i], p).grad.detach())
                        temp.append(torch.std(grad[i][n_params_single:]).item()**2) # Variances of the objective terms 
                    min_var = min(temp)

                    for i in range(obj_nb-1):  
                        lambda_[i] = sqrt(min_var/temp[i]) 
                    lambda_list[n,:] = lambda_
                
            if n_params_single is not None : 
                print('Lambdas', lambda_, 'Inverse Param', torch.exp(params[:n_params_single]))          
            else : 
                print('Lambdas', lambda_)
            
            ham = hamiltonian(params, momentum, log_prob_func, obj_nb, lambda_, sampler=sampler, inv_mass=inv_mass)
            leapfrog_params, leapfrog_momenta, grad_ll = leapfrog(params, momentum, log_prob_func, obj_nb, lambda_, sampler=sampler, steps=num_steps_per_sample, step_size=step_size, inv_mass=inv_mass, store_on_GPU = store_on_GPU, debug=debug)
            
            for i in range(num_steps_per_sample): 
                ham_list[n, i] = hamiltonian(leapfrog_params[i].to(device).detach().requires_grad_(), leapfrog_momenta[i].to(device), log_prob_func, obj_nb, lambda_, sampler=sampler, inv_mass=inv_mass)
            
            
            params = leapfrog_params[-1].to(device).detach().requires_grad_()
            momentum = leapfrog_momenta[-1].to(device)
            new_ham = hamiltonian(params, momentum, log_prob_func, obj_nb, lambda_, sampler=sampler, inv_mass=inv_mass)

            leap_params.append(torch.stack(leapfrog_params).detach())
            leap_momentum.append(torch.stack(leapfrog_momenta).detach())

            grad_list[:,:,:,n] = grad_ll 
                              
            # ======================================= Metropolis Hasting Step =========================================
            rho = min(0., acceptance(ham, new_ham))
            if debug == 1:
                print('Step: {:.2f}, Current Hamiltoninian: {:.2f}, Proposed Hamiltoninian: {:.2f}, Momentum : {:.2f}, Potentiel: {:.2f}'.format(n,ham,new_ham, 0.5*torch.dot(momentum, momentum).item(), (new_ham-0.5*torch.dot(momentum, momentum)).item()))
                
            if rho >= torch.log(torch.rand(1)): 
                if debug == 1:
                    print('Accept') 
                
                if n > burn:
                    if store_on_GPU:
                        ret_params.append(leapfrog_params[-1]) 
                    else:
                        ret_params.append(leapfrog_params[-1].cpu())
                else: 
                    if store_on_GPU:
                        ret_params = [leapfrog_params[-1].clone()] 
                    else:
                        ret_params = [leapfrog_params[-1].clone().cpu()]
                    
            else:
                num_rejected += 1 
                AR.append(n)
                params = ret_params[-1].to(device) 

                if n > burn:
                    if store_on_GPU:
                        ret_params.append(ret_params[-1].to(device))
                    else:
                        ret_params.append(ret_params[-1].cpu())
                if debug == 1:
                    print('Reject')
                    
            if NUTS and n <= burn:
                desired_accept_rate = 0.8
                if n < burn:
                    step_size, eps_bar, H_t = adaptation(rho, n, step_size_init, H_t, eps_bar, desired_accept_rate=desired_accept_rate)
                if n  == burn:
                    step_size = eps_bar
                    print('Final Adapted Step Size: ', step_size)
     
            if n >= burn :
                log_prob = 0
                for i in range(obj_nb):
                    log_prob += log_prob + lambda_[i]*log_prob_func(params)[i]
                log_prob_list.append((log_prob).detach())
                
                  
        except util.LogProbError:
            num_rejected += 1
            params = ret_params[-1].to(device)
            if n > burn:
                if store_on_GPU:
                    ret_params.append(ret_params[-1].to(device))
                else:
                    ret_params.append(ret_params[-1].cpu())
            if debug == 1:
                print('REJECT')
            

        if not store_on_GPU: 
            momentum = None; leapfrog_params = None; leapfrog_momenta = None; ham = None; new_ham = None

            del momentum, leapfrog_params, leapfrog_momenta, ham, new_ham
            torch.cuda.empty_cache()
    
    util.progress_bar_end('Acceptance Rate {:.2f}'.format(1 - num_rejected/num_samples)) 

    return list(map(lambda t: t.detach(), ret_params)), leap_params, leap_momentum, ham_list, grad_list, log_prob_list, AR, lambda_list
    
    
