# Description: This script is used to generate the data for the SW network with IID coupling. The script is parallelized over the different values of R and v2. The script generates the data for the coexistence and the maximum derivative of the density for the SW network with IID coupling. The script saves the data in the folder data/revision_network_IID/SW_IID/SW_iid_parallel_T-<tmax>/.

import numpy as np
import networkx as nx
import time
import scipy.integrate as integrate
from numbalsoda import lsoda
from multiprocessing import Pool, cpu_count
import os

import sys
sys.path.append('../modules')
import model as mod
import model_pnas as mod_ss

S = 5
N = 100
p = 0.2
nn = 2
cvec = np.ones((S,N))

farray = np.linspace(0.5,2,S)
xi = 1/N

# x0_array = np.linspace(0.1,2.0,50)
# var_array = np.linspace(0.01,2.5,50)

# R_array = x0_array*np.exp(var_array/2)
# v2_array = (np.exp(var_array)-1)*R_array**2


R_array = np.linspace(0.01,1.5,50)
v2_array = np.linspace(0.01,10,50)

# x0_array = (R_array**2/np.sqrt(v2_array + R_array**2))
# var_array = np.log(1+v2_array/R_array**2)

rho0 = np.random.uniform(0,1/S,(S,N))
tmax = 100000

def loop_over_var(params):

    CUTSTART_T = time.time()
    i,kernels,seed = params
    np.random.seed(seed)
    R = R_array[i]
    across_var_coex = np.zeros(len(v2_array))
    across_var_max_der = np.zeros(len(v2_array))
    for j in range(len(v2_array)):
        x0 = R**2/np.sqrt(v2_array[j] + R**2)
        sig2 = np.log(1+v2_array[j]/R**2)
        rarray = np.random.lognormal(np.log(x0), np.sqrt(sig2), size = (S, N))
        earray = np.zeros((S, N), dtype = np.float64)

        for k in range(S):
            earray[k,:] = np.mean(kernels[k,:,:]) * N / rarray[k,:]

        t_eval = np.linspace(0, tmax, 500)

        # rho_soln = integrate.solve_ivp(mod_ss.rhodot_function_solveivp, [0, tmax], rho0.flatten(), args = (kernels, cvec, earray,S,N), method = 'RK45')
        # rho_er_ms = rho_soln.y.reshape((S,N,int(rho_soln.t.size)))
        # across_var_coex[j] = np.where(np.mean(rho_er_ms[:,:,-1], axis = 1)>1e-5)[0].size
        # across_var_max_der[j] =  np.max(np.abs(mod_ss.rhodot_function_solveivp(0,rho_er_ms[:,:,-1].flatten(), kernels, cvec, earray,S,N)))

        # del rho_soln,rho_er_ms

        rhodot_function_solveivp_lsoda = mod_ss.make_lsoda_func_rhodot(kernels, cvec, earray, S, N)
        funcptr = rhodot_function_solveivp_lsoda.address
        usol_temp,success_temp = lsoda(funcptr, rho0.flatten(), np.linspace(0,0.1,10))
        usol, success = lsoda(funcptr, rho0.flatten(), t_eval)
        rho_er_ms = usol.reshape((int(t_eval.size),S,N))
        across_var_coex[j] = np.where(np.mean(rho_er_ms[-1,:,:], axis = 1)>1e-5)[0].size
        across_var_max_der[j] =  np.max(np.abs(mod_ss.rhodot_function_solveivp(0,rho_er_ms[-1,:,:].flatten(), kernels, cvec, earray,S,N)))

        del usol,rho_er_ms

        
    # print(i, time.time()-CUTSTART_T)
    return across_var_coex, across_var_max_der

def iterator_x0(kernels,randomSeeds):
    iterable = [(i,kernels,randomSeeds[i]) for i in range(R_array.size)]
    with Pool(cpu_count()) as p:
        results = p.map(loop_over_var, iterable)
    coex_array = np.zeros((R_array.size, v2_array.size), dtype = np.float64)
    maxder = np.zeros((R_array.size, v2_array.size), dtype = np.float64)
    for i in range(len(results)):
        coex_array[i,:] = results[i][0]
        maxder[i,:] = results[i][1]

    return coex_array, maxder

if __name__ == '__main__':
    randomSeeds = np.random.randint(0,1000000,R_array.size)
    print(time.ctime())

    repNetworks = 25
    dataFolder =  "../data/revision_network_IID/SM_IID/"
    os.makedirs(dataFolder+"SM_iid_parallel_T-%d"%tmax,exist_ok=True)

    for repID in range(repNetworks):
        START_TIME = time.time()

        sm_network = nx.watts_strogatz_graph(N,nn,p)
        while(nx.is_connected(sm_network) == False):
            sm_network = nx.watts_strogatz_graph(N,nn,p)
            
        kernels = np.zeros((S, N, N), dtype = np.float64)

        for i in range(S):
            f = farray[i]    
            sm_kernel = mod_ss.find_effective_kernel(f, xi, sm_network)
            kernels[i,:,:] = sm_kernel

        coex_array, maxder = iterator_x0(kernels,randomSeeds)
        np.savez_compressed(dataFolder+"SM_iid_parallel_T-%d/rep_%d.npz"%(tmax,repID), coex_array = coex_array, maxder = maxder, R_array = R_array, v2_array = v2_array)
        print(repID,time.time()-START_TIME,"seconds")

    print(time.ctime())