# Description: Script to compute the stationary density of the OCN model on a RGG network.

import numpy as np
import matplotlib.pyplot as plt

from numba import njit, prange

import sys
sys.path.append('../modules')
import model as mod
import utils

import networkx as nx

local = False

if local:
    sys.path.append('../../lib/')
    import funPlots as fplot

    fplot.master_format(ncols = 1, nrows = 1)

from matplotlib.colors import LinearSegmentedColormap
import scipy.integrate as integrate
import matplotlib.colors as mcolors


import time as measure_time


def ricker_wavelet(x, p1, p2, m = 3):
    return (1 - (m*x)**2/p1**2)*np.exp(-(m*x)**2/(2*p2**2))

def add_transparency_to_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # Get the colormap
    original_cmap = plt.get_cmap(cmap)
    
    # Get the colors from the original colormap
    colors = original_cmap(np.linspace(minval, maxval, n))
    
    # Add an alpha channel based on the y values
    colors[:, 3] = np.linspace(0, 1, n)
    
    # Create the new colormap
    new_cmap = mcolors.LinearSegmentedColormap.from_list(cmap + "_transparent", colors)
    
    return new_cmap

def decay_fun(x, a1 = 2, a2 = 6, a3 = 3):
    return a1*np.exp(-a2*x**2) - (a1 - 1)*np.exp(-a3*x**2)

def generate_RGG(N, d, seed, plot = False):
    np.random.seed(seed)
    pos_x = np.random.uniform(0, 1, N)
    pos_y = np.random.uniform(0, 1, N)
    pos = dict(zip(range(N), zip(pos_x, pos_y)))

    RGG_net = nx.random_geometric_graph(N, d, pos = pos)

    pos_mat = np.array(list(pos.values()))
    dist_mat = utils.euclidean_distance_matrix(pos_mat)

    mindistance = np.min(dist_mat[dist_mat > 0])
    for (u, v) in RGG_net.edges():
        RGG_net[u][v]['weight'] = mindistance/dist_mat[u, v]

    if plot:
        plt.figure(figsize = (6, 6))
        nx.draw(RGG_net, pos = pos, node_size = 100, node_color = 'k', width = 0.5, edge_color = 'k')
        plt.show()

    return RGG_net, pos, dist_mat

def generate_covmat_rw(dist_mat, sigma_diag, p1, p2, m = 3):
    covmat = ricker_wavelet(dist_mat, p1, p2, m)
    covmat[np.diag_indices_from(covmat)] = sigma_diag

    min_eig = np.min(np.real(np.linalg.eigvals(covmat)))

    if min_eig < 0:
        raise ValueError("Covariance matrix is not positive semi-definite")
    
    return covmat


N = 100
S = 5

RGG_net, pos, dist_mat = generate_RGG(N, d = 0.17, seed = 12, plot = False)


p1 = 1.3
p2_array = np.linspace(0.05, 1, 20)

rho0 = np.ones((S,N))*0.5
cvec = np.ones((S,N))

f_array = np.ones(S)
xi = 1

sigma_diag = 1

kernels = np.zeros((S, N, N), dtype = np.float64)
for alpha in range(S):
    kernels[alpha] = mod.find_effective_kernel(f_array[alpha], xi, RGG_net)

tmax = 10000

mu = 0.5
mu_array = np.ones(N)*mu

Nrep = 500

rhotot_all = []


rhotot_all = list(rhotot_all)
for idx_rep in range(Nrep):
    tic = measure_time.time()
    print(f"Repetition {idx_rep + 1} of {Nrep}")

    rhotot = np.zeros(p2_array.size)
    for idx, p2 in enumerate(p2_array):
        covmat = generate_covmat_rw(dist_mat, sigma_diag, p1, p2, m = 3)

        r = np.zeros((N, S))
        e = np.zeros((S, N))

        for alpha in range(S):
            r[:, alpha] = np.exp(np.random.multivariate_normal(mu_array, covmat))
            e[alpha] = np.mean(kernels[alpha]) * N / r[:, alpha]

        rho_soln = integrate.solve_ivp(mod.rhodot_function_solveivp, [0, tmax], rho0.flatten(), args = (kernels, cvec, e, S, N), method = 'RK45')
        rho_er_ms = rho_soln.y.reshape((S, N, int(rho_soln.t.size)))
        rhostat = rho_er_ms[:,:,-1]

        rhotot[idx] = np.sum(rhostat)
    rhotot_all.append(rhotot)
    toc = measure_time.time()
    print(f"\t Time elapsed: {toc - tic:.2f} seconds")

rhotot_all = np.array(rhotot_all)

np.save(f"../data/RGG_OCN/rhostat_RGG_N{N}_S{S}_Nrep{Nrep}_p1{p1}_mu{mu}.npy", rhotot_all)
np.save(f"../data/RGG_OCN/p2array_RGG_N{N}_S{S}_Nrep{Nrep}_p1{p1}_mu{mu}.npy", p2_array)