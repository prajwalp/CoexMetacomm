# Description: We generate the data for the general model for Nreplicas replicas. We save the data in the folder data/general_data_plots.

import numpy as np

import sys
sys.path.append('../modules')
import model as mod

import time as measure_time

path = "../data/general_data_plots/"

N = 100000
Nreplicas = 100
S = 20

th = 1e-6

R_array = np.linspace(0.5, 4, 50)
vsq_array = np.linspace(0.5, 60, 60)

Delta_array = np.linspace(-2.5, 2.5, S)

np.save(path + "R_array_N" + str(N) + "_S" + str(S) + ".npy", R_array)
np.save(path + "vsq_array_N" + str(N) + "_S" + str(S) + ".npy", vsq_array)
np.save(path + "Delta_array_N" + str(N) + "_S" + str(S) + ".npy", Delta_array)

for idx_replica in range(Nreplicas):
    t0 = measure_time.time()
    print("Replica ", idx_replica)

    coex, npatch_dom, part_rate_loc, spread_loc = mod.find_properties_general(N, S, Delta_array, R_array, vsq_array, th)

    np.save(path + "coex_general_N" + str(N) + "_S" + str(S) + "_replica" + str(idx_replica) + ".npy", coex)
    np.save(path + "npatch_dom_general_N" + str(N) + "_S" + str(S) + "_replica" + str(idx_replica) + ".npy", npatch_dom)
    np.save(path + "part_rate_loc_general_N" + str(N) + "_S" +  str(S) + "_replica" + str(idx_replica) + ".npy", part_rate_loc)
    np.save(path + "spread_loc_general_N" + str(N) + "_S" + str(S) + "_replica" + str(idx_replica) + ".npy", spread_loc)

    t1 = measure_time.time()
    print("\t Elapsed time: ", np.round(t1 - t0,2), " seconds")
    print()
