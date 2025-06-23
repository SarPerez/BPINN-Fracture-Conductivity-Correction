import torch
import hamiltorch
import numpy as np

def build_lists(models, n_params_single=None, tau_priors=None, mu_priors = None, tau_likes=0.1, pde = False):

    if n_params_single is not None:
        n_params = [n_params_single]
    else:
        n_params = []

    if isinstance(tau_priors,list) or tau_priors is None:
        build_tau_priors = False
    else:
        build_tau_priors = True
        tau_priors_elt = tau_priors
        tau_priors = []

    if isinstance(mu_priors,list) or mu_priors is None:
        build_mu_priors = False
    else:
        build_mu_priors = True
        mu_priors_elt = mu_priors
        mu_priors = []

    if isinstance(tau_likes,list):
        build_tau_likes = False
    else:
        build_tau_likes = True
        tau_likes_elt = tau_likes
        tau_likes = []

    params_shape_list = []
    params_flattened_list = []

    if n_params_single is not None:
        for _ in range(n_params_single):
            params_flattened_list.append(1)
            params_shape_list.append(1)
            if build_tau_priors:
                tau_priors.append(tau_priors_elt)
            if build_mu_priors: 
                mu_priors.append(mu_priors_elt)

    for model in models:
        print(model)
        n_params.append(hamiltorch.util.flatten(model).shape[0])
        if build_tau_likes:
            tau_likes.append(tau_likes_elt)
        for weights in model.parameters():
            params_shape_list.append(weights.shape)
            params_flattened_list.append(weights.nelement())
            if build_tau_priors:
                tau_priors.append(tau_priors_elt)
            if build_mu_priors: 
                mu_priors.append(mu_priors_elt)

    # if we deal with pde then we also have data of residual
    if pde and build_tau_likes:
        tau_likes.append(tau_likes_elt)
    
    n_params = list(np.cumsum(n_params)) 
    
    return params_shape_list, params_flattened_list, n_params, tau_priors, mu_priors, tau_likes


def define_model_log_prob_bpinns(models, model_loss, data, tau_priors=None, mu_priors = None, tau_likes=None, predict=False, n_params_single = None, pde = False):

    """This function defines the `log_prob_func` for torch nn.Modules. This will then be passed into the hamiltorch sampler. This is an important
    function for any work with Bayesian neural networks.

    Parameters
    ----------
    models : list of torch.nn.Module(s)
        This is the list of torch neural network models, which will be used when performing inference.
    model_loss : function
        This determines the likelihood to be used for the model. You can customize this function in main code.
    data : dictionary
        Training input output data of each model.
    tau_priors: float or list of float(s)
        Determines the stds of gaussian priors for parameters. If this is None then the priors become uniform distribution. If this is float then it becomes std of priors for all parameters. If this is a list then each element of the list becomes std of priors for [1st single parameter, 2nd single parameter,..., weights of 1st hidden layer, bias of 1st hidden layer, weights of 2nd hidden layer, bias of 2nd hidden layer,...]
    mu_priors: float or list of float(s)
        Determines the means of gaussian priors for parameters. If this is None then the priors become centered in zeros. If this is float then it becomes means of priors for all parameters. If this is a list then each element of the list becomes means of priors for [1st single parameter, 2nd single parameter,..., weights of 1st hidden layer, bias of 1st hidden layer, weights of 2nd hidden layer, bias of 2nd hidden layer,...]
    tau_likes: float or list of float(s)
        Data are assumed to be collected with gaussian noise and tau_likes determines the std of noise. If this is float then it becomes std of noise for all data. If this is a list then each element of the list becomes std of noise for each element of the list of models.
    predict : bool
        Flag to set equal to `True` when used as part of `hamiltorch.predict_model`, otherwise set to False. This controls the number of objects to return.
    n_params_single : int
        The number of single parameters that have to be inferred.
    pde : bool
        Determines whether there is pde or not.

    Returns
    -------
    function
        Returns a `log_prob_func`, which takes a 1-D torch.tensor of a length equal to the parameter dimension.
        It returns a single value for the prediction, and the list of the objective terms for the sampling.

    """
    _, params_flattened_list, n_params, tau_priors, mu_priors, tau_likes = build_lists(models, n_params_single, tau_priors, mu_priors, tau_likes, pde)
        
    fmodel = [] 
    for model in models:
        fmodel.append(hamiltorch.util.make_functional(model)) 
        
    if tau_priors is not None and mu_priors is not None:
        dist_list = []
        for (tau, mu) in zip(tau_priors, mu_priors): 
            dist_list.append(torch.distributions.Normal(mu, tau**-0.5)) 
    elif tau_priors is not None :
        dist_list = []
        for tau in tau_priors: 
            dist_list.append(torch.distributions.Normal(0, tau**-0.5))
            
    def log_prob_func(params):
        
        params_unflattened = []
        if n_params_single is not None:
            params_single = params[:n_params[0]]
            for i in range(len(models)):
                params_unflattened.append(hamiltorch.util.unflatten(models[i], params[n_params[i]:n_params[i+1]]))
        else:
            params_single = None
            for i in range(len(models)):
                if i == 0:
                    params_unflattened.append(hamiltorch.util.unflatten(models[i], params[:n_params[i]]))
                else:
                    params_unflattened.append(hamiltorch.util.unflatten(models[i] ,params[n_params[i-1]:n_params[i]]))

        l_prior = torch.zeros_like(params[0], requires_grad=True) 
        
        if tau_priors is not None:
            i_prev = 0
            norm_ = 0
            for index, dist in zip(params_flattened_list, dist_list):
                w = params[i_prev:index+i_prev]
                norm_ += torch.dot((w-dist.loc), (w-dist.loc))
                l_prior = -norm_/(2*dist.scale**2)
                i_prev += index

        def gradients(outputs, inputs):
            return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)
      
        ll_list, output = model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single)
        
        if predict : 
            obj_nb = len(ll_list) # number of objectives in the model_loss function (excluding the prior)
            ll = 0 
            for i in range(obj_nb):
                ll += ll_list[i]   
            ll += l_prior
            return ll, output # return a single value for the prediction
        else :
            ll_list.append(l_prior)
            return ll_list # return a list of objectif terms for the sampling 
                
    return log_prob_func


def sample_model_bpinns(models, data, model_loss, num_samples=10, num_steps_per_sample=10, step_size=0.1, burn=-1, num_adap_step = 20, inv_mass=None, sampler=hamiltorch.Sampler.AW_HMC, debug=False, tau_priors=None, mu_priors = None, tau_likes=None, store_on_GPU = True, device = 'cpu', n_params_single = None, pde = False, params_init_val = None):

    """Sample weights from a NN model to perform inference. This function builds a `log_prob_func` from the torch.nn.Module and passes it to `hamiltorch.sample`.

    Parameters
    ----------
    models : list of torch.nn.Module(s)
        This is the list of torch neural network models, which will be used when performing inference.
    data : dictionary
        Training input output data of each model.
    model_loss : function
        This determines the likelihood to be used for the model. You can customize this function in main code.
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
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC, Sampler.RMHMC, Sampler.HMC_NUTS}.        
    debug : {0, 1}
        Debug mode can take 2 options. Setting debug = 0 (default) allows the sampler to run as normal. Setting debug = 1 prints both the old and new Hamiltonians per iteration.
    tau_priors: float or list of float(s)
        Determines the stds of gaussian priors for parameters. If this is None then the priors become uniform distribution. If this is float then it becomes std of priors for all parameters. If this is a list then each element of the list becomes std of priors for [1st single parameter, 2nd single parameter,..., weights of 1st hidden layer, bias of 1st hidden layer, weights of 2nd hidden layer, bias of 2nd hidden layer,...]
    mu_priors: float or list of float(s)
        Determines the means of gaussian priors for parameters. If this is None then the priors become centered in zeros. If this is float then it becomes means of priors for all parameters. If this is a list then each element of the list becomes means of priors for [1st single parameter, 2nd single parameter,..., weights of 1st hidden layer, bias of 1st hidden layer, weights of 2nd hidden layer, bias of 2nd hidden layer,...]
    tau_likes: float or list of float(s)
        Data are assumed to be collected with gaussian noise and tau_likes determines the std of noise. If this is float then it becomes std of noise for all data. If this is a list then each element of the list becomes std of noise for each element of the list of models.
    store_on_GPU : bool
        Option that determines whether to keep samples in GPU memory. It runs fast when set to TRUE but may run out of memory unless set to FALSE.
    device : name of device, or {'cpu', 'cuda'}
        The device to run on.
    n_params_single : int
        The number of single parameters that have to be inferred.
    pde : bool
        Determines whether it is pde or not.
    params_init_val : torch.tensor
        Initialisation of the parameters.
        Shape (d,) where d = D with D the dimensionality of the parameters, or d < D (only the inverse parameters e.g.).

    Returns
    -------
    param_samples : list of torch.tensor(s)
        A list of parameter samples. The full trajectory will be returned such that selecting the proposed params requires indexing [1::L] to remove params_innit and select the end of the trajectories.
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

    if n_params_single is not None: 
        params_init = torch.zeros([n_params_single]).to(device)
    else:
        params_init = torch.Tensor([]).to(device)

    for model in models:
        params_init = torch.cat((params_init, hamiltorch.util.flatten(model).to(device).clone() )) 

    if params_init_val is not None: 
        n_init_val = params_init_val.shape[0] # number of initial values for the parameters (forward and inverse pb)
        if n_init_val < params_init.shape[0]:
            # here we can pass initial values only for some parameters (for the n_params_single e.g.)
            params_init = torch.cat((params_init_val, params_init[n_init_val:]))
        else:
            # or for the whole set of parameters (in case of preconditioning)
            params_init = params_init_val
            
    log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, mu_priors, tau_likes, n_params_single = n_params_single, pde = pde)
    
    lambda_ = [1.0]*(len(tau_likes)+1) # Init of the lambda values for the weighted approach (if Sampler.HMC the lambdas remain at this default value)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return hamiltorch.sample(log_prob_func, params_init, lambda_, num_samples=num_samples, num_steps_per_sample=num_steps_per_sample, step_size=step_size, burn=burn, num_adap_step = num_adap_step, n_params_single = n_params_single, inv_mass=inv_mass, sampler=sampler, debug=debug, store_on_GPU = store_on_GPU)

def predict_model_bpinns(models, samples, data, model_loss, tau_priors=None, mu_priors = None, tau_likes=None, lambda_list = None, n_params_single = None, pde = False):

    """Function used to make predictions given model samples. 
    
    Parameters
    ----------
    models : list of torch.nn.Module(s)
        This is the list of torch neural network models, which will be used when performing inference.
    samples : list of torch.tensor(s)
        A list, where each element is a torch.tensor of shape (D,), where D is the number of parameters of the model. The length of the list is given by the number of samples, S.
    data : dictionary
        Training input output data of each model.
    model_loss : function
        This determines the likelihood to be used for the model. You can customize this function in main code.
    tau_priors: float or list of float(s)
        Determines the stds of gaussian priors for parameters. If this is None then the priors become uniform distribution. If this is float then it becomes std of priors for all parameters. If this is a list then each element of the list becomes std of priors for [1st single parameter, 2nd single parameter,..., weights of 1st hidden layer, bias of 1st hidden layer, weights of 2nd hidden layer, bias of 2nd hidden layer,...]
    mu_priors: float or list of float(s)
        Determines the means of gaussian priors for parameters. If this is None then the priors become centered in zeros. If this is float then it becomes means of priors for all parameters. If this is a list then each element of the list becomes means of priors for [1st single parameter, 2nd single parameter,..., weights of 1st hidden layer, bias of 1st hidden layer, weights of 2nd hidden layer, bias of 2nd hidden layer,...]
    tau_likes: float or list of float(s)
        Data are assumed to be collected with gaussian noise and tau_likes determines the std of noise. If this is float then it becomes std of noise for all data. If this is a list then each element of the list becomes std of noise for each element of the list of models.
    lambda_list : list
        List of the weights for each objective term in the potential energy (including the priors) and for each adaptive iteration in the AW_HMC algorithm.
    n_params_single : int
        The number of single parameters that have to be inferred.
    pde : bool
        Determines whether it is pde or not.

    Returns
    -------
    predictions : torch.tensor
        Output of the model of shape (S,N,O), where S is the number of samples, N is the number of data points, and O is the output shape of the model.
    """
    for i in range(len(tau_likes)): 
        tau_likes[i] = tau_likes[i]*lambda_list[i] # Update of the tau_likes values for the prediction given the lambda weights 
        
    if pde:

        log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, mu_priors, tau_likes, predict=True, n_params_single = n_params_single, pde = pde)

        pred_log_prob_list = []
        pred_list = []
        _, pred = log_prob_func(samples[0])
        for i in range(len(pred)):
            pred_list.append([])

        for s in samples:
            lp, pred = log_prob_func(s)
            pred_log_prob_list.append(lp.detach()) 
            for i in range(len(pred_list)):
                pred_list[i].append(pred[i].detach())

        for i in range(len(pred_list)):
            pred_list[i] = torch.stack(pred_list[i])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred_list

    else:
        with torch.no_grad():

            log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, mu_priors, tau_likes, predict=True, n_params_single = n_params_single, pde = pde)
            
            pred_log_prob_list = []
            pred_list = []
            
            _, pred = log_prob_func(samples[0])
            for i in range(len(pred)):
                pred_list.append([])
                
            for s in samples:
                lp, pred = log_prob_func(s)
                pred_log_prob_list.append(lp.detach()) 
                for i in range(len(pred_list)):
                    pred_list[i].append(pred[i].detach()) 

            for i in range(len(pred_list)):
                pred_list[i] = torch.stack(pred_list[i])
                    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return pred_list

