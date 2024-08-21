# Description: Script to generate the stationary distribution of the OCN model for high and low correlation

import numpy as np
import networkx as nx
import scipy.integrate as integrate

import sys
sys.path.append('../modules/')
import model 
import OCNfun
import utils

def ricker_wavelet(x, p1, p2, m = 3):
    return (1 - (m*x)**2/p1**2)*np.exp(-(m*x)**2/(2*p2**2))

def decay_fun(x, a1 = 2, a2 = 6, a3 = 3):
    return a1*np.exp(-a2*x**2) - (a1 - 1)*np.exp(-a3*x**2)

def generate_covmat_rw(dist_mat, sigma_diag, p1, p2, m = 3):
    covmat = ricker_wavelet(dist_mat, p1, p2, m)
    covmat[np.diag_indices_from(covmat)] = sigma_diag

    min_eig = np.min(np.real(np.linalg.eigvals(covmat)))

    if min_eig < 0:
        raise ValueError("Covariance matrix is not positive semi-definite, min eigenvalue = {}".format(min_eig))
    
    return covmat

dimX = 64
dimY = 64

cellSize = 1

outletSide = "N"
outletPos = 17

adj, E, A, Zmat, OCNlandscape = OCNfun.createOCN(dimX, dimY, outletPos, outletSide, cellSize,
                                                 seed = 49, slope = 1, undirected = False,
                                                 return_adj = True)
adj_OCN, pos_OCN, OCN_mask = OCNfun.get_aggregated_OCN(OCNlandscape, thrA = 10)
G = nx.from_numpy_array(adj_OCN)

np.save("../data/RGG_OCN/adj_OCN.npy", adj_OCN)
np.save("../data/RGG_OCN/pos_OCN.npy", pos_OCN)
np.save("../data/RGG_OCN/Zmat_OCN.npy", Zmat)
np.save("../data/RGG_OCN/Areas_OCN.npy", A[np.where(OCN_mask.flatten() != 0)])

OCN_net = nx.from_numpy_array(adj_OCN)
dist_mat_spat_OCN = utils.euclidean_distance_matrix(pos_OCN/np.sqrt(dimX*dimY))

mindistance_OCN = np.min(dist_mat_spat_OCN[dist_mat_spat_OCN > 0])
for (u, v) in OCN_net.edges():
    OCN_net[u][v]['weight'] = mindistance_OCN/dist_mat_spat_OCN[u, v]

dist_mat_net_OCN = np.array(nx.floyd_warshall_numpy(OCN_net))

N_OCN = adj_OCN.shape[0]

S = 5
f_array = np.ones(S)
xi = 1

sigma_diag_OCN = 1.5

p1_OCN = 5
p2_array = np.linspace(0.05, 1, 20)

tmax = 10000
mu = 1
mu_array = np.ones(N_OCN)*mu

covmat_OCN = generate_covmat_rw(dist_mat_net_OCN, sigma_diag_OCN, p1_OCN, p2_array[-1], m = 0.8)

rho0_OCN = np.ones((S, N_OCN))*0.5
cvec_OCN = np.ones((S, N_OCN))

kernels_OCN = np.zeros((S, N_OCN, N_OCN), dtype = np.float64)
for alpha in range(S):
    kernels_OCN[alpha] = model.find_effective_kernel(f_array[alpha], xi, OCN_net)

tmax = 5000
covmat_OCN = generate_covmat_rw(dist_mat_net_OCN, sigma_diag_OCN, p1_OCN, p2_array[-1], m = 0.8)
np.random.seed(301)

r_OCN = np.zeros((N_OCN, S))
e_OCN = np.zeros((S, N_OCN))
for alpha in range(S):
    r_OCN[:, alpha] = np.exp(np.random.multivariate_normal(mu_array, covmat_OCN))
    e_OCN[alpha] = kernels_OCN[alpha].mean() * N_OCN / r_OCN[:, alpha]

rho_soln_OCN = integrate.solve_ivp(model.rhodot_function_solveivp, [0, tmax], rho0_OCN.flatten(), args = (kernels_OCN, cvec_OCN, e_OCN, S, N_OCN), method = 'RK45')
rho_er_ms_OCN = rho_soln_OCN.y.reshape((S, N_OCN, int(rho_soln_OCN.t.size)))
Time_OCN = rho_soln_OCN.t

rhostat_OCN = rho_er_ms_OCN[:,:,-1]
coex_OCN = np.where(rhostat_OCN.mean(axis = -1) > 0.1)[0].size

np.save(f"../data/RGG_OCN/rhostat_OCN_highcorr_mu{mu}_p1{p1_OCN}.npy", rhostat_OCN)


covmat_OCN = generate_covmat_rw(dist_mat_net_OCN, sigma_diag_OCN, p1_OCN, p2_array[0], m = 0.8)

rho0_OCN = np.ones((S, N_OCN))*0.5
cvec_OCN = np.ones((S, N_OCN))

kernels_OCN = np.zeros((S, N_OCN, N_OCN), dtype = np.float64)
for alpha in range(S):
    kernels_OCN[alpha] = mod.find_effective_kernel(f_array[alpha], xi, OCN_net)

np.random.seed(351)

r_OCN = np.zeros((N_OCN, S))
e_OCN = np.zeros((S, N_OCN))

for alpha in range(S):
    r_OCN[:, alpha] = np.exp(np.random.multivariate_normal(mu_array, covmat_OCN))
    e_OCN[alpha] = kernels_OCN[alpha].mean() * N_OCN / r_OCN[:, alpha]

rho_soln_OCN = integrate.solve_ivp(model.rhodot_function_solveivp, [0, tmax], rho0_OCN.flatten(), args = (kernels_OCN, cvec_OCN, e_OCN, S, N_OCN), method = 'RK45')
rho_er_ms_OCN = rho_soln_OCN.y.reshape((S, N_OCN, int(rho_soln_OCN.t.size)))
Time_OCN = rho_soln_OCN.t

rhostat_OCN = rho_er_ms_OCN[:,:,-1]

coex_OCN = np.where(rhostat_OCN.mean(axis = -1) > 0.1)[0].size

np.save(f"../data/RGG_OCN/rhostat_OCN_lowcorr_mu{mu}_p1{p1_OCN}.npy", rhostat_OCN)
