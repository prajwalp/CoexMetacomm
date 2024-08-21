import numpy as np
from numba import njit, cfunc,carray,prange
import scipy.optimize as optimize
from numbalsoda import lsoda_sig
import utils
import networkx as nx
import matplotlib.pyplot as plt

############################
# COMPUTE EFFECTIVE KERNEL #
############################

@njit
def find_effective_kernel_nb(f, xi, adj_matrix, Laplacian = None,
                             L_eigvals = None, V = None, V_inv = None,
                             undirected = True):
    """
    Finds the effective coupling between different patches assuming a Monod
    function for the explorer creation rate. Numba boosted.

    Note that this is already the transpose kernel appearing as is in the
    dynamics.

    Parameters
    ----------
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
    adj_matrix : numpy.ndarray
        Adjacency matrix of the network.
    Laplacian : numpy.ndarray
        Laplacian matrix of the network. Must be of type np.complex128.
        If None, it is computed from the network.
    L_eigvals : numpy.ndarray
        Eigenvalues of the Laplacian matrix. Must be of type np.complex128.
        If None, they are computed from the Laplacian matrix.
    V : numpy.ndarray
        Eigenvectors of the Laplacian matrix. Must be of type np.complex128.
        If None, they are computed from the Laplacian matrix.
    V_inv : numpy.ndarray
        Inverse of the eigenvectors of the Laplacian matrix. Must be of type
        np.complex128.
        If None, they are computed from the Laplacian matrix.

    Returns
    -------
    effective_coupling : numpy.ndarray
        Effective coupling between patches.
    """
    N = adj_matrix.shape[0]

    if Laplacian is None:
        Laplacian = utils.find_laplacian_nb(adj_matrix)
        Laplacian = Laplacian.astype(np.complex128)

    if L_eigvals is None or V is None or V_inv is None:
        if undirected:
            L_eigvals, V, V_inv = utils.diagonalize_matrix_sym(Laplacian.T)
            V = V.astype(np.complex128)
            V_inv = V_inv.astype(np.complex128)
            L_eigvals = L_eigvals.astype(np.complex128)
        else:
            L_eigvals, V, V_inv = utils.diagonalize_matrix(Laplacian.T)

    w_matrix = utils.create_diag_matrix(1/(1+f*L_eigvals))

    cNormCoupling = np.real(np.dot(V, np.dot(w_matrix, V_inv)))

    effective_coupling = np.dot(cNormCoupling, adj_matrix.T)*xi*f/(1+f)

    return effective_coupling

def find_effective_kernel(f, xi, network, Laplacian = None,
                          L_eigvals = None, V = None, V_inv = None,
                          undirected = True):
    """
    Wrapper for find_effective_kernel_nb.

    Parameters
    ----------
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
    network : networkx.Graph
        Network of connected patches.
    Laplacian : numpy.ndarray
        Laplacian matrix of the network.
        If None, it is computed from the network.
    L_eigvals : numpy.ndarray
        Eigenvalues of the Laplacian matrix.
        If None, they are computed from the Laplacian matrix.
    V : numpy.ndarray
        Eigenvectors of the Laplacian matrix.
        If None, they are computed from the Laplacian matrix.
    V_inv : numpy.ndarray
        Inverse of the eigenvectors of the Laplacian matrix.
        If None, they are computed from the Laplacian matrix.
    undirected : bool
        If True, the network is considered undirected.
    q_regular : bool
        If True, the network is considered q-regular, where q is the degree.

    Returns
    -------
    effective_coupling : numpy.ndarray
        Effective coupling between patches.
    """

    adj_matrix = nx.adjacency_matrix(network).toarray().astype(float)

    if Laplacian is not None:
        Laplacian = Laplacian.astype(np.complex128)

    if L_eigvals is not None or V is not None or V_inv is not None:
        V = V.astype(np.complex128)
        V_inv = V_inv.astype(np.complex128)
        L_eigvals = L_eigvals.astype(np.complex128)

    K = find_effective_kernel_nb(f, xi, adj_matrix, Laplacian, L_eigvals, V, V_inv,
                                 undirected = undirected)

    return K


@njit
def find_metapopulation_capacity_nb(K, adj, f, xi, undirected = True):
    """
    Compute the metapopulation capacity of a network.
    If the network has more than one connected component, the metapopulation
    capacity of each component is computed and the minimum is returned.

    Parameters
    ----------
    K : numpy.ndarray
        Effective coupling matrix.
    adj : numpy.ndarray
        Adjacency matrix of the network.
        Used to check if the network is connected.
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
    undirected : bool
        If True, the network is considered undirected.

    Returns
    -------
    lambdaMax : float
        Metapopulation capacity.
    """
    CComponents = utils.find_connected_components(adj)
    NComponents = len(CComponents)

    if NComponents == 1:
        return np.max(np.real(np.linalg.eigvals(K)))
    else:
        lambdaMax_components = np.zeros(NComponents, dtype = np.float64)

        for i in range(NComponents):
            module = CComponents[i]
            adj_sub = utils.extract_submatrix(adj, module)
            Ktemp = find_effective_kernel_nb(f, xi, adj_sub,
                                             undirected = undirected)
            lambdaMax_components[i] = np.max(np.real(np.linalg.eigvals(Ktemp)))

        return np.min(lambdaMax_components)

@njit
def find_connected_metapopulation_capacity_nb(K):
    """
    Compute the metapopulation capacity of a network. This function assumes that
    the network is connected, so use it only if you are sure that the network has
    only one connected component.

    Parameters
    ----------
    K : numpy.ndarray
        Effective coupling matrix.

    Returns
    -------
    lambdaMax : float
        Metapopulation capacity.
    """
 
    return np.max(np.real(np.linalg.eigvals(K)))


def find_metapopulation_capacity(K, network, f, xi, undirected = True,
                                 guaranteed_connected = False):
    """
    Wrapper for find_metapopulation_capacity_nb.

    Parameters
    ----------
    K : numpy.ndarray
        Effective coupling matrix.
    network : networkx.Graph
        Network of connected patches.
        Needed to check if the network is connected.
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
    undirected : bool
        If True, the network is considered undirected.
    guaranteed_connected : bool
        If True, the network is considered connected.
        No check is performed.

    Returns
    -------
    lambdaMax : float
        Metapopulation capacity.
    """
    adj_matrix = nx.adjacency_matrix(network).toarray().astype(float)

    K = K.astype(np.complex128)

    if guaranteed_connected:
        lambdaMax = find_connected_metapopulation_capacity_nb(K)
    else:
        lambdaMax = find_metapopulation_capacity_nb(K, adj_matrix, f, xi,
                                                    undirected = undirected)
    
    return lambdaMax


################################
# SIMULATE DYNAMICAL EVOLUTION #
################################

def make_lsoda_func_rhodot(K,cvec,evec,S,N):
    """
    Creates the function to be passed to the lsoda solver for the dynamical
    evolution of the system.
    
    Parameters
    ----------
    K : numpy.ndarray
        Effective coupling between patches.
    cvec : numpy.ndarray
        Initial concentration of resources in each patch.
    evec : numpy.ndarray
        Efficiency of explorers in each patch.
    S : int
        Number of patches.
    N : int
        Number of resources.
        
    Returns
    -------
    rhodot_function_solveivp_lsoda : function
        Function to be passed to the lsoda solver."""
    
    @cfunc(lsoda_sig)
    def rhodot_function_solveivp_lsoda(t, x, dx, p):
        """
        Function to be passed to the lsoda solver for the dynamical evolution
        of the system.
        
        Parameters
        ----------
        t : float
            Time.
        x : numpy.ndarray
            Current state of the system.
        dx : numpy.ndarray
            Derivative of the current state of the system.
        p : tuple
            Tuple containing the parameters of the system.
            
        Returns
        -------
        dx_ : numpy.ndarray
            Derivative of the current state of the system.
        """
        
        x_ = carray(x, (S*N,))
        dx_ = carray(dx, (S*N,))
        rho = x_.reshape((S,N))
        drho = dx_.reshape((S,N))
        sumrho = np.sum(rho,axis=0)
        for i in range(S):
            if(np.mean(rho[i]) < 1e-9):
                drho[i] = 0
            else:
                drho[i] = -rho[i]*evec[i] + (1 - sumrho)*np.dot(K[i], rho[i]*cvec[i])
        dx_ = drho.flatten()
        
    return rhodot_function_solveivp_lsoda

@njit
def rhodot_function_solveivp(t,x,K,cvec,evec,S,N):
    """
    Function to be passed to the solve_ivp solver for the dynamical evolution
    of the system.

    Parameters
    ----------
    t : float
        Time.
    x : numpy.ndarray
        Current state of the system.
    K : numpy.ndarray
        Effective coupling between patches.
    cvec : numpy.ndarray
        Initial concentration of resources in each patch.
    evec : numpy.ndarray
        Efficiency of explorers in each patch.
    S : int
        Number of patches.
    N : int
        Number of resources.

    Returns
    -------
    drho.flatten() : numpy.ndarray
        Derivative of the current state of the system.
    """

    rho = x.reshape((S,N))
    drho = np.zeros(rho.shape)
    sumrho = np.sum(rho,axis=0)
    for i in range(S):
        if(np.mean(rho[i]) < 1e-9):
            drho[i] = 0
        else:
            drho[i] = -rho[i]*evec[i] + (1 - sumrho)*np.dot(K[i], rho[i]*cvec[i])
            drho[i][drho[i]**2<=1e-14] = 0
    return drho.flatten()

@njit
def simulate_S(S, N, Nsteps, dt, K,  cvec, evec, rho0, epsvec = None,               
             check_stationary = False, tol=1e-8):
    """
    Simulates the model for a given set of parameters. Arbitrary number of species.

    Parameters
    ----------
    N : int
        Number of patches.
    S : int
        Number of species.
    Nsteps : int
        Number of time steps.
    dt : float
        Time step.
    K : numpy.ndarray
        array of Effective kernel between patches. Dimensions must be (S, N, N).
    cvec : numpy.ndarray
        Vector of creation rates of explorers.
    evec : numpy.ndarray
        Vector of death rates of settled population.
    epsvec : numpy.ndarray
        Vector of "pathogen" self-inhibition term. 
    rho0 : numpy.ndarray
        Initial condition for the density of settled population in each patch.
        If None, it is set to a random vector.
    check_stationary : bool
        If True, the simulation is stopped when the density of settled population
        is stationary.        
        
    Returns
    -------
    rho : numpy.ndarray
        Density of settled population in each patch at each time step.
    """

    rho = np.zeros((Nsteps, S, N), dtype=np.float64)
    if epsvec is None:
        epsvec = np.zeros((S, N))

    rho[0, :, :] = rho0


    for t in range(Nsteps - 1):
        for s in range(S):
            if((rho[t][s] > 1e-8).all()):
                rho[t+1][s] = rho[t][s] + dt*(-rho[t][s]*(evec[s] + epsvec[s] * rho[t][s]) + 
                                      (1 - np.sum(rho[t], axis=0))*np.dot(K[s], rho[t][s]*cvec[s]))
            else:
                rho[t+1][s] = rho[t][s]
        
        # if check_stationary:
        #     if np.abs(np.sum(rho[t+1] - rho[t])) < tol:
        #         for i in range(N):
        #             for s in range(S):
        #                 rho[t+1:, s, i] = rho[t+1, s, i]
        #         break
    return rho

def find_statpop_escan_2S_nb(e1arr, e2arr, adj_matrix1, adj_matrix2, Nsteps, dt, f1, f2, xi1, xi2,
                       rho0 = None,
                       epsvec = None,
                       campl = 0.0,
                       undirected = True):                       
    """
    Compute the stationary population for a range of f and xi values.

    Parameters
    ----------
    e1arr: numpy.ndarray
        Array of e values first pop
    e2arr : numpy.ndarray
        Array of e values second pop
    network : networkx.Graph
        Networkx graph of the network.
    Nsteps : int
        Number of time steps.
    dt : float
        Time step.
    f1 : numpy.ndarray
        Exploration parameter first pop
    f2 : numpy.ndarray
        Exploration parameter second pop
    rho0 : numpy.ndarray
        Initial condition for the density of settled population in each patch.        
    undirected : bool
        If True, the network is considered undirected.

    Returns
    -------
    statpop : numpy.ndarray
        Array of stationary population ratios \sum_i rho1 / rho2
    """

    
    N = adj_matrix1.shape[0]
    K = np.zeros((2, N, N), dtype=np.float64)
    adj_matrices = np.stack((adj_matrix1, adj_matrix2))
    xis = (xi1, xi2)
    fs = (f1, f2)
    for idx, adj_matrix in enumerate(adj_matrices):
        Laplacian = utils.find_laplacian_nb(adj_matrix)
        Laplacian = Laplacian.astype(np.complex128)    
        if undirected:
            L_eigvals, V, V_inv = utils.diagonalize_matrix_sym(Laplacian.T)
            L_eigvals = L_eigvals.astype(np.complex128)
            V = V.astype(np.complex128)
            V_inv = V_inv.astype(np.complex128)
        else:
            L_eigvals, V, V_inv = utils.diagonalize_matrix(Laplacian.T)
        K[idx, :, :] = find_effective_kernel_nb(fs[idx], xis[idx], adj_matrix, Laplacian,                                        
                                         L_eigvals, V, V_inv,
                                         undirected = undirected)
    
    
    # K2 = find_effective_kernel_nb(f2, xi2, adj_matrix, Laplacian,
                                        #  L_eigvals, V, V_inv,
                                        #  undirected = undirected)
    # 
    statpop = np.zeros((e1arr.size, e2arr.size, 2, N), dtype = np.float64)
    cvec = np.stack((np.full(N, 1.0), np.full(N, 1.0))) 
    if rho0 is None:
        rho0 = np.random.random((2, N))            
    for idx_e1 in prange(e1arr.size):
        e1 = e1arr[idx_e1]        
        for idx_e2 in range(e2arr.size):
            e2 = e2arr[idx_e2]
            evec = np.stack((np.full(N, e1), np.full(N, e2)))            
            rho = simulate_S(2, N, Nsteps, dt, K, cvec, evec, rho0, epsvec)
            #check that all rho have reached a constant value
            # if np.abs(rho[-1] - rho[-2]).max() > 1e-3:
            #     print("Warning: simulation has not reached a steady state.")
            #     print("Stopping simulation, run with more timesteps")
                
            #     return statpop

            statpop[idx_e1, idx_e2] = rho[-1]
            

    return statpop

def find_statpop_escan_2S(e1arr, e2arr, network1, network2, NSteps, dt, f1, f2, xi1, xi2, rho0 = None,
                       epsvec = None,
                       cvec_noise = 0.0,
                       undirected = True):                       
    """
    Compute the stationary population for a range of f and xi values.

    Parameters
    ----------
    e1arr: numpy.ndarray
        Array of e values first pop
    e2arr : numpy.ndarray
        Array of e values second pop
    network : networkx.Graph
        Networkx graph of the network.
    Nsteps : int
        Number of time steps.
    dt : float
        Time step.
    f1 : numpy.ndarray
        Exploration parameter first pop
    f2 : numpy.ndarray
        Exploration parameter second pop
    rho0 : numpy.ndarray
        Initial condition for the density of settled population in each patch.        
    undirected : bool
        If True, the network is considered undirected.

    Returns
    -------
    statpop : numpy.ndarray
        Array of stationary population ratios \sum_i rho1 / rho2
    """

    adj_matrix_1 = nx.adjacency_matrix(network1).toarray().astype(float)    
    adj_matrix_2 = nx.adjacency_matrix(network2).toarray().astype(float)    
    if rho0 is None:
        rho0 = np.full((2, adj_matrix_1.shape[0]), 1e-3)
    N = adj_matrix_1.shape[0]
    if epsvec is None:
        epsvec = np.zeros((2, N))
    statpop = find_statpop_escan_2S_nb(e1arr, e2arr, adj_matrix_1, adj_matrix_2,
                                NSteps, dt, f1, f2, xi1, xi2, rho0, epsvec,
                                cvec_noise,
                                undirected = undirected)

    return statpop




def solnfunc(x, r, N):
    """
    Consistency condition for the average density of each species,
    assuming that that r follows a given probability distribution.

    Parameters
    ----------
    x : array
        Array of average densities per species.
    r : array
        Array of r, which is the ratio between the mean-field kernel
        for each species and the death rate for said species in a
        given patch.
    N : int
        Number of patches.

    Returns
    -------
    float
        Consistency condition for the average density of each species.
    """

    rbpb = r@x
    avg = r.T@(1/(1+rbpb)) / N
    return avg - 1

def solnfunc_nonumba(x, r, N):
    """
    Consistency condition for the average density of each species,
    assuming that that r follows a given probability distribution.

    Parameters
    ----------
    x : array
        Array of average densities per species.
    r : array
        Array of r, which is the ratio between the mean-field kernel
        for each species and the death rate for said species in a
        given patch.
    N : int
        Number of patches.

    Returns
    -------
    float
        Consistency condition for the average density of each species.
    """

    rbpb = r@x
    avg = r.T@(1/(1+rbpb)) / N
    return np.sum((avg - 1)**2)

def solve_pavg(r, verbose = False, ftol = 1e-15, gtol = 1e-15, xtol = 1e-15):
    """
    Finds the average of each species with a random realization of r, optimizing
    the consistency condition for the average density of each species.

    Parameters
    ----------
    r : array
        Array of r, which is the ratio between the mean-field kernel
        for each species and the death rate for said species in a
        given patch.
    verbose : bool, optional
        If True, prints the output of the optimization. The default is False.
    ftol : float, optional
        Tolerance for termination by the change of the cost function. The default is 1e-15.
    gtol : float, optional
        Tolerance for termination by the norm of the gradient. The default is 1e-15.
    xtol : float, optional
        Tolerance for termination by the change of the independent variables. The default is 1e-15.

    Returns
    -------
    pavg : array
        Array of average densities per species.
    """

    N, S = r.shape

    if verbose:
        v = 2
    else:
        v = 0

    p0 = np.random.random(S)

    scipysoln = optimize.least_squares(solnfunc, p0, bounds=(0,1),
                                       args=(r, N),
                                       ftol = ftol,
                                       gtol = gtol,
                                       xtol = xtol,
                                       verbose = v)
    
    return scipysoln.x

@njit
def solnfunc_heterogenous(x, K, e,S,N):
    """
    Consistency condition for the average density of each species,
    assuming that that r follows a given probability distribution.

    Parameters
    ----------
    x : array
        Array of average densities per species.
    K : array
        Array of dispersal kernels for each species.
    e : array
        Array of death rates for each species.
    S : int
        Number of species.
    N : int
        Number of patches.

    Returns
    -------
    float
        Consistency condition for the average density of each species.
    """
    rho = np.reshape(x, (S, N))
    rhosoln = np.zeros((S, N), dtype = np.float64)
    sumrho = np.sum(rho,axis=0)
    for i in range(S):
        Ki = K[i, :, :]
        rhoi = rho[i]
        K_rhoi = Ki@rhoi
        ei = e[i]
        rhosoln[i, :]  = -ei*rhoi + (1-sumrho)*K_rhoi

    return rhosoln.flatten()

def solve_p_heterogenous(e,K, verbose = False, ftol = 3e-16, gtol = 3e-16, xtol = 3e-16,max_nfev=int(1e6)):
    """
    Finds the average of each species with a random realization of r, optimizing
    the consistency condition for the average density of each species.

    Parameters
    ----------
    r : array
        Array of r, which is the ratio between the mean-field kernel
        for each species and the death rate for said species in a
        given patch.
    verbose : bool, optional
        If True, prints the output of the optimization. The default is False.
    ftol : float, optional
        Tolerance for termination by the change of the cost function. The default is 1e-15.
    gtol : float, optional
        Tolerance for termination by the norm of the gradient. The default is 1e-15.

    Returns
    -------
    pavg : array
        Array of average densities per species.
    """

    S = K.shape[0]
    N = K.shape[1]

    if verbose:
        v = 2
    else:
        v = 0

    p0 = np.random.random((S,N)).flatten()

    scipysoln = optimize.least_squares(solnfunc_heterogenous, p0, bounds=(0,1),
                                       args=(K,e,S,N),
                                       ftol = ftol,
                                       gtol = gtol,
                                       xtol = xtol,
                                       verbose = v,
                                       max_nfev=max_nfev)
    
    return scipysoln.x

@njit
def find_Q(r, pavg):
    """
    Finds the density of empty space in each patch, given the average density
    of each species.

    Parameters
    ----------
    r : array
        Array of r, which is the ratio between the mean-field kernel
        for each species and the death rate for said species in a
        given patch.
    pavg : array
        Array of average densities per species.

    Returns
    -------
    Q : array
        Array of density of empty space in each patch.
    """

    return 1/(1+r@pavg)

@njit
def find_p_from_avg(r, pavg):
    """
    Finds the density of each species in each patch, given the average density
    of each species.

    Parameters
    ----------
    r : array
        Array of r, which is the ratio between the mean-field kernel
        for each species and the death rate for said species in a
        given patch.
    pavg : array
        Array of average densities per species.

    Returns
    -------
    p : array
        Array of density of each species in each patch.
    """

    Q = find_Q(r, pavg)
    p = Q[..., None] * pavg[None, ...]
    return p*r

def find_p(r, verbose = False, ftol = 1e-15, gtol = 1e-15, xtol = 1e-15):
    """
    Finds the density of each species in each patch, given a random realization
    of r.

    Parameters
    ----------
    r : array
        Array of r, which is the ratio between the mean-field kernel
        for each species and the death rate for said species in a
        given patch.
    verbose : bool, optional
        If True, prints the output of the optimization. The default is False.
    ftol : float, optional
        Tolerance for termination by the change of the cost function. The default is 1e-15.
    gtol : float, optional
        Tolerance for termination by the norm of the gradient. The default is 1e-15.
    xtol : float, optional
        Tolerance for termination by the change of the independent variables. The default is 1e-15.

    Returns
    -------
    p : array
        Array of density of each species in each patch.
    """

    pavg = solve_pavg(r, verbose = verbose, ftol = ftol, gtol = gtol, xtol = xtol)
    return find_p_from_avg(r, pavg)

def check_coex(p, th = 1e-10):
    """
    Check how many species coexist in the system, given the density of each
    species in each patch.

    Parameters
    ----------
    p : array
        Array of density of each species in each patch.
    th : float, optional
        Threshold for the density of each species. The default is 1e-10.

    Returns
    -------
    Ncoex : int
        Number of coexisting species.
    """

    pmax = p.max(axis = 0)
    Ncoex = np.sum(pmax > th)

    return Ncoex

def find_properties_IID(N, S, x0_array, sigmasq_array, th,
                        verbose = False, ftol = 1e-15,
                        gtol = 1e-15, xtol = 1e-15):
    """

    Parameters
    ----------
    N : int
        Number of patches.
    S : int
        Number of species.
    x0_array : array
        Array of x0 values. The mean of the lognormal distribution is
        x0 exp(sigma^2/2).
    sigmasq_array : array
        Array of sigma^2 values.
    th : float
        Threshold for the density of each species.
    verbose : bool, optional
        If True, prints the output of the optimization. The default is False.
    ftol : float, optional
        Tolerance for termination by the change of the cost function. The default is 1e-15.
    gtol : float, optional
        Tolerance for termination by the norm of the gradient. The default is 1e-15.
    xtol : float, optional
        Tolerance for termination by the change of the independent variables. The default is 1e-15.
    
    Returns
    -------
    """
    
    coex = np.zeros((len(x0_array), len(sigmasq_array)))
    npatch_dom = np.zeros((len(x0_array), len(sigmasq_array), 4, S))
    part_rate_loc = np.zeros((len(x0_array), len(sigmasq_array), S))
    spread_loc = np.zeros((len(x0_array), len(sigmasq_array), S))

    for idx_x0, x0 in enumerate(x0_array):

        if verbose:
            print("Computing x0 = ", np.round(x0,2))

        for idx_sigma, sigmasq in enumerate(sigmasq_array):
            sigma = np.sqrt(sigmasq)
            r = np.random.lognormal(np.log(x0), sigma, size=(N, S))
            p = find_p(r, ftol = ftol, gtol = gtol, xtol = xtol)

            for idx_dom, th_dom in enumerate([0.5, 0.7, 0.9, 0.95]):
                npatch_dom[idx_x0, idx_sigma, idx_dom] = np.sum(p > th_dom, axis = 0)

            part_rate_loc[idx_x0, idx_sigma] = np.sum(p**4, axis = 0)/np.sum(p**2, axis = 0)**2*N
            coex[idx_x0, idx_sigma] = check_coex(p, th = th)

            for s in range(S):
                vals = np.where(np.cumsum(np.sort(p[:, s])) > p[:,s].sum()/2)[0]
                if len(vals) != 0:
                    spread_loc[idx_x0, idx_sigma, s] = N - vals[0]
            
    return coex, npatch_dom, part_rate_loc, spread_loc


def find_PD_IID(N, S, x0_array, sigmasq_array, th,
                verbose = False, ftol = 1e-15,
                gtol = 1e-15, xtol = 1e-15):
    """
    Evaluates the number of coexisting species as a function of x0 and sigma^2,
    for a given number of patches N and a given number of species S. The result
    is a phase diagram to be compared with the coexistence transition, that is
    <r> > 1.
    The distribution of r is IID and taken to be lognormal.

    Parameters
    ----------
    N : int
        Number of patches.
    S : int
        Number of species.
    x0_array : array
        Array of x0 values. The mean of the lognormal distribution is
        x0 exp(sigma^2/2).
    sigmasq_array : array
        Array of sigma^2 values.
    th : float
        Threshold for the density of each species.
    verbose : bool, optional
        If True, prints the output of the optimization. The default is False.
    ftol : float, optional
        Tolerance for termination by the change of the cost function. The default is 1e-15.
    gtol : float, optional
        Tolerance for termination by the norm of the gradient. The default is 1e-15.
    xtol : float, optional
        Tolerance for termination by the change of the independent variables. The default is 1e-15.
    
    Returns
    -------
    coex : array
        Array of number of coexisting species.
    """
    
    coex = np.zeros((len(x0_array), len(sigmasq_array)))

    for idx_x0, x0 in enumerate(x0_array):

        if verbose:
            print("Computing x0 = ", np.round(x0,2))

        for idx_sigma, sigmasq in enumerate(sigmasq_array):
            sigma = np.sqrt(sigmasq)
            r = np.random.lognormal(np.log(x0), sigma, size=(N, S))
            p = find_p(r, ftol = ftol, gtol = gtol, xtol = xtol)
            coex[idx_x0, idx_sigma] = check_coex(p, th = th)
    
    return coex

def plot_coex_IID_N(N_PD, S_array, x0_array, sigmasq_array, cmap_PD, savepath = None):
    """
    Plot the number of coexisting species as a function of x0 and sigma^2, for 
    a given number of patches N_PD and a given number of species S_array.
    The simulation is assumed to be already performed and the data is loaded from
    the data folder. The distribution of r is IID and taken to be lognormal.

    Parameters
    ----------
    N_PD : int
        Number of patches.
    S_array : array
        Array of number of species.
    x0_array : array
        Array of x0 values.
    sigmasq_array : array
        Array of sigma^2 values.
    cmap_PD : colormap
        Colormap for the number of coexisting species.
    savepath : str
        Path where to save the figure. If None, the figure is not saved.

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
        Figure and axes objects of the plot.    
    """
    fig, ax = plt.subplots(2, S_array.size//2, figsize=(25, 7))
    ival = S_array.size//2

    for idx_S, S_PD in enumerate(S_array):

        try:
            PD = np.load(f'../data/PD_IID/coex_IID_N{N_PD}_S{S_PD}.npy')
        except:
            continue

        PD = np.load(f'../data/PD_IID/coex_IID_N{N_PD}_S{S_PD}.npy')
        
        aax = ax[idx_S//ival, idx_S%ival]

        im = aax.pcolormesh(x0_array, sigmasq_array, PD.T, cmap= cmap_PD, shading='auto')
        aax.set_xlabel('$x_0$', labelpad = -5, fontsize = 20)
        aax.set_ylabel('$\sigma^2$', labelpad = -5, fontsize = 20)
        aax.set_title(f'$S = {S_PD}, N = {N_PD}$', fontsize = 15)
        aax.plot(x0_array[x0_array < 1], -2*np.log(x0_array[x0_array < 1]), color='darkred', lw = 2.5, ls = '--')
        aax.set_ylim(sigmasq_array.min(), sigmasq_array.max())
        aax.set_xticks(np.round(np.linspace(x0_array.min(), x0_array.max(), 4), 2))
        aax.set_yticks(np.round(np.linspace(sigmasq_array.min(), sigmasq_array.max(), 5), 2))
        aax.axvline(x = 1, color = 'black', lw = 2.5, ls = '--', alpha = 0.7)

        # colorbar
        cbar = plt.colorbar(im, ax=aax)
        cbar.set_label('N. of coexisting species', rotation=270, labelpad=15, fontsize = 16)
        cbar.set_ticks(np.linspace(0, S_PD, 2).astype(int))
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    if savepath is not None:
        plt.savefig(savepath + f'coex_N{N_PD}_S{S_PD}.png', dpi = 300, bbox_inches='tight')
    
    return fig, ax


def find_pvals_general(N, S, Delta_min, Delta_max, R_array, vsq_array,
                       verbose = False, ftol = 1e-15,
                       gtol = 1e-15, xtol = 1e-15):
    """
    Finds the average, maximum and minimum density of each species in each patch,
    given a random realization of r. The distribution of r is IID and taken to be
    lognormal.

    Parameters
    ----------
    N : int
        Number of patches.
    S : int
        Number of species.
    Delta_min : float
        Minimum value of Delta.
    Delta_max : float
        Maximum value of Delta.
    R_array : array
        Array of R values.
    vsq_array : array
        Array of v^2 values.
    verbose : bool, optional
        If True, prints the output of the optimization. The default is False.
    ftol : float, optional
        Tolerance for termination by the change of the cost function. The default is 1e-15.
    gtol : float, optional
        Tolerance for termination by the norm of the gradient. The default is 1e-15.
    xtol : float, optional
        Tolerance for termination by the change of the independent variables. The default is 1e-15.

    Returns
    -------
    pavg : array
        Array of average densities per species.
    pmax : array
        Array of maximum densities per species.
    pmin : array
        Array of minimum densities per species.
    Delta_array : array
        Array of Delta values.
    """
    
    Delta_array = np.random.uniform(Delta_min, Delta_max, size = S)
    pavg = np.zeros((len(R_array), len(vsq_array), S))
    pmax = np.zeros((len(R_array), len(vsq_array), S))
    pmin = np.zeros((len(R_array), len(vsq_array), S))

    for idx_R, R in enumerate(R_array):

        if verbose:
            print("Computing R = ", np.round(R,2))

        for idx_v, vsq in enumerate(vsq_array):
            sigmasq_array = np.log(1 + vsq/(R + Delta_array/S)**2)
            mu_array = (R + Delta_array/S)*np.exp(-sigmasq_array/2)

            r = np.zeros((N, S), dtype = np.float64)

            for alpha in range(S):
                r[:, alpha] = np.random.lognormal(np.log(mu_array[alpha]), np.sqrt(sigmasq_array[alpha]), size = N)

            p = find_p(r, ftol = ftol, gtol = gtol, xtol = xtol)
            
            pavg[idx_R, idx_v, :] = p.mean(axis = 0)
            pmax[idx_R, idx_v, :] = p.max(axis = 0)
            pmin[idx_R, idx_v, :] = p.min(axis = 0)

    return pavg, pmax, pmin, Delta_array
    

def find_properties_general(N, S, Delta_array, R_array, vsq_array, th,
                            verbose = False, ftol = 1e-15,
                            gtol = 1e-15, xtol = 1e-15):
    """
    Finds the number of coexisting species, the number of patches dominated by
    each species, the partitioning rate and the spread of each species, given
    a random realization of r. The distribution of r is IID and taken to be
    lognormal.

    Parameters
    ----------
    N : int
        Number of patches.
    S : int
        Number of species.
    Delta_array : array
        Array of Delta values.

    Returns
    -------
    coex : array
        Array of number of coexisting species.
    npatch_dom : array  
        Array of number of patches dominated by each species.
    part_rate_loc : array
        Array of partitioning rate of each species.
    spread_loc : array
        Array of spread of each species.
    """
   
    coex = np.zeros((len(R_array), len(vsq_array)))
    npatch_dom = np.zeros((len(R_array), len(vsq_array), 4, S))
    part_rate_loc = np.zeros((len(R_array), len(vsq_array), S))
    spread_loc = np.zeros((len(R_array), len(vsq_array), S))

    for idx_R, R in enumerate(R_array):

        if verbose:
            print("Computing R = ", np.round(R, 2))

        for idx_v, vsq in enumerate(vsq_array):
            sigmasq_array = np.log(1 + vsq/(R + Delta_array/S)**2)
            mu_array = (R + Delta_array/S)*np.exp(-sigmasq_array/2)

            r = np.zeros((N, S), dtype = np.float64)

            for alpha in range(S):
                r[:, alpha] = np.random.lognormal(np.log(mu_array[alpha]), np.sqrt(sigmasq_array[alpha]), size = N)

            p = find_p(r, ftol = ftol, gtol = gtol, xtol = xtol)

            for idx_dom, th_dom in enumerate([0.1, 0.2, 0.5, 0.9]):
                npatch_dom[idx_R, idx_v, idx_dom] = np.sum(p > th_dom, axis = 0)

            part_rate_loc[idx_R, idx_v] = np.sum(p**4, axis = 0)/np.sum(p**2, axis = 0)**2*N
            coex[idx_R, idx_v] = check_coex(p, th = th)

            for s in range(S):
                vals = np.where(np.cumsum(np.sort(p[:, s])) > p[:,s].sum()/2)[0]
                if len(vals) != 0:
                    spread_loc[idx_R, idx_v, s] = N - vals[0]
            
    return coex, npatch_dom, part_rate_loc, spread_loc


def plot_coex_general_N(N_PD, S_array, R_array, vsq_array, Delta_min, Delta_max,
                        cmap_PD, savepath = None, th = 1e-3, fig = None, ax = None):
    """
    Plot the number of coexisting species as a function of R and v^2, for
    a given number of patches N_PD and a given number of species S_array.
    The simulation is assumed to be already performed and the data is loaded from
    the data folder. The distribution of r is IID and taken to be lognormal.

    Parameters
    ----------
    N_PD : int
        Number of patches.
    S_array : array
        Array of number of species.
    R_array : array
        Array of R values.
    vsq_array : array
        Array of v^2 values.
    Delta_min : float
        Minimum value of Delta.
    Delta_max : float
        Maximum value of Delta.
    cmap_PD : colormap
        Colormap for the number of coexisting species.
    savepath : str
        Path where to save the figure. If None, the figure is not saved.
    th : float, optional
        Threshold for the density of each species. The default is 1e-3.
    fig : matplotlib figure object, optional
        Figure object where to plot the data. If None, a new figure is created. The default is None.
    ax : matplotlib axes object, optional
        Axes object where to plot the data. If None, a new axes is created. The default is None.

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
        Figure and axes objects of the plot.
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, S_array.size, figsize=(25, 3))

    path = "../data/pvals_general/"

    for idx_S, S_PD in enumerate(S_array):

        try:
            pmax = np.load(path + f'N{N_PD}_S{S_PD}_pmax_Delta{Delta_min}_{Delta_max}.npy')
            Delta_array = np.load(path + f'N{N_PD}_S{S_PD}_Delta_Delta{Delta_min}_{Delta_max}.npy')
        except:
            continue

        PD = np.sum(pmax > th, axis = -1)
        
        aax = ax[idx_S]

        im = aax.pcolormesh(R_array, vsq_array, PD.T, cmap= cmap_PD, shading='auto',
                            vmin = 0, vmax = S_PD)
        aax.set_xlabel('$R$', labelpad = -5, fontsize = 20)
        aax.set_ylabel('$v^2$', labelpad = -5, fontsize = 20)
        aax.set_title(f'$S = {S_PD}, N = {N_PD}$', fontsize = 15)
        
        xx = R_array[R_array > 1]
        yy = np.mean(Delta_array) - np.min(Delta_array)
        yy *= xx**2/(xx - 1)


        aax.plot(xx, yy, color='darkred', lw = 2.5, ls = '--')
        aax.set_ylim(vsq_array.min(), vsq_array.max())
        aax.set_xticks(np.round(np.linspace(R_array.min(), R_array.max(), 4), 2))
        aax.set_yticks(np.round(np.linspace(vsq_array.min(), vsq_array.max(), 5), 2))
        aax.axvline(x = 1, color = 'black', lw = 2.5, ls = '--', alpha = 0.7)

        # colorbar
        cbar = plt.colorbar(im, ax=aax)
        cbar.set_label('N. of coexisting species', rotation=270, labelpad=20, fontsize = 16)
        cbar.set_ticks(np.linspace(0, S_PD, 2).astype(int))
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    if savepath is not None:
        plt.savefig(savepath + f'coex_general_N{N_PD}_S{S_PD}.png', dpi = 300, bbox_inches='tight')
    
    return fig, ax
