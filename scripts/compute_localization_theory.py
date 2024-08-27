# Description: computes the localization measure for a given set of S species and a specific set of parameters by evaluating numerically eq. (13).
# Output is stored as two lists of S elements and saved in JSON format. The first list is the x vector defined in eq.(10), which represents the rescaled average population of all species. The second list contains the localization measure eq. (13).
# Evaluation of the functions is computationally heavy for a large number of species.

#! /usr/bin/env python3

import numpy as np
import scipy.optimize as optimize
import mpmath as mp

import json
import argparse
# import mkl
# mkl.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--exec_name", type=str)
parser.add_argument("--include_exec_tstamp", action="store_true", default=False)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--output_fname_root", type=str)
parser.add_argument("--output_fname_ext", type=str)
parser.add_argument("--output_fname_full", type=str)

parser.add_argument("--delta_alpha", type=float, nargs="+", required=True, help="List containing Delta_alpha, deviation of each species' baseline fitness from average, see definition in the paper")
parser.add_argument("--Nspec", type=int, required=True, help="Number of species, must be equal to the length of delta_alpha")
parser.add_argument("--R", type=float, required=True, help="Baseline fitness of all species, see definition in the paper")
parser.add_argument("--v", type=float, required=True, help="Landscape heterogeneity parameter, see definition in the paper")


args, unknown = parser.parse_known_args()
sim_output_dir = args.output_dir


def lognorm_mp(r, x_alpha, sigma_alpha):    
    return mp.exp(-(mp.log(r) - mp.log(x_alpha))**2/(2*sigma_alpha**2)) / r / mp.sqrt(2*mp.pi*sigma_alpha**2)


def Wln(z, x0, sigma):
    res = mp.quad(lambda x : lognorm_mp(x, x0, sigma) * mp.exp(-x * z), [0, mp.inf])
    return res


def Fbar(z, x, R, v, deltas):
    S = len(deltas)
    term_two = 0
    for alpha in range(S):
        sigma_alpha_sq = mp.log(1.0 + v**2/(R+deltas[alpha]/S)**2)
        a_alpha = (R + deltas[alpha] / S) * mp.exp(-sigma_alpha_sq / 2)
        term_two += mp.log(Wln(z*x[alpha], a_alpha, mp.sqrt(sigma_alpha_sq)))
    return z - term_two / S


def Wln_prime_int(z, k, x0, sigma):
    res = mp.quad(lambda r : (-r)**k * lognorm_mp(r, x0, sigma) * mp.exp(-r * z), [0, mp.inf])
    return res


def get_x_zero_order(R, v, deltas):    
    return R * ((deltas - np.mean(deltas)) / v**2 + (R - 1) / R**2)


def integrand_xalpha(z, x, R, v, deltas, alpha):
    S = mp.mpf(len(deltas))
    sigma_alpha_sq = mp.log(1.0 + v**2/(R+deltas[alpha]/S)**2)
    a_alpha = (R + deltas[alpha] / S) * mp.exp(-sigma_alpha_sq / 2)
    return S * mp.exp(-S*Fbar(z, x, R, v, deltas)) *(-Wln_prime_int(z*x[alpha], mp.mpf(1), a_alpha, mp.sqrt(sigma_alpha_sq)) / Wln(z*x[alpha], a_alpha, mp.sqrt(sigma_alpha_sq)))


def get_integral(x, R, v, deltas, alpha):    
    S = len(x)
    x_mp = mp.matrix(S,1)
    deltas_mp = mp.matrix(S,1)
    
    for q in range(S):
        x_mp[q] = mp.mpf(x[q])        
        deltas_mp[q] = mp.mpf(deltas[q])
    
    integral = mp.quadts( lambda z : integrand_xalpha(z, x_mp, mp.mpf(R), mp.mpf(v), deltas, alpha), [0, mp.inf])
   
    return np.float64(integral)


def thisF(x, R, v, deltas):
    S = len(deltas)
    res = np.empty(S, dtype=np.float64)
    for alpha in range(S):
        res[alpha] = np.abs(get_integral(x, R, v, deltas, alpha) - 1)
    return res


def get_x_th(R, v, deltas, S):
    guess = get_x_zero_order(R, v, deltas)
    guess = np.where(guess>0, guess, 0).astype(np.float128)    
    guess = np.where(guess<S, guess, S).astype(np.float128)    
    print("guess", guess / S )                   
    res = optimize.least_squares(thisF, guess, args=(R,v,deltas), bounds=(0,S))                                
    return res.x


def integrand_localpha(z, x, R, v, deltas, alpha, n):
    S = mp.mpf(len(deltas))
    sigma_alpha_sq = mp.log(1.0 + v**2/(R+deltas[alpha]/S)**2)
    a_alpha = (R + deltas[alpha] / S) * mp.exp(-sigma_alpha_sq / 2)
    return z**(n-1) * mp.exp(-S*Fbar(z, x, R, v, deltas)) *(Wln_prime_int(z*x[alpha], mp.mpf(n), a_alpha, mp.sqrt(sigma_alpha_sq)) / Wln(z*x[alpha], a_alpha, mp.sqrt(sigma_alpha_sq)))

def get_nth_mom_p(x, n,  R, v, deltas, alpha):    
    integral = mp.quadts( lambda z : integrand_localpha(z, x, R, v, deltas, alpha, n), [0, mp.inf])    
    return x[alpha]**(n)  / np.math.factorial(n) * np.float64(integral)

def get_inv_part_ratio_th(R, v, deltas, provided_x):
    S = len(deltas)
    res = np.empty(S)
    for alpha in range(S):
        p_4 = get_nth_mom_p(provided_x, 4, R, v, deltas, alpha)
        p_2 = get_nth_mom_p(provided_x, 2, R, v, deltas, alpha)        
        res[alpha] = np.nan_to_num(p_4 / (p_2)**2)
    return res


deltas = args.delta_alpha
S = args.Nspec
R = args.R
v = args.v

assert S == len(deltas)
output = np.zeros((2, S))

output[0, :] = get_x_th(R, v, deltas, S)
print(output[0, :])
output[1, :] = get_inv_part_ratio_th(R, v, deltas, output[0, :])
print(output[1, :])

json.dump(output.tolist(), open(sim_output_dir + "/" + args.output_fname_full, "wt"))
