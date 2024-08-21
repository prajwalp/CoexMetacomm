import numpy as np
import networkx as nx

import utils

import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

import time as measure_time

base = importr('base')
OCNet = importr('OCNet')
igraphR = importr('igraph', robject_translations={'.env': '__env'})
dollar = base.__dict__["$"]

def createOCN(dimX, dimY, outletPos, outletSide, cellSize,
              seed = 42, slope = 0.01, undirected = False,
              return_adj = False):
    """
    Creates an Optimal Channel Network (OCN) using the OCNet R package.
    The OCN is created as a spanning tree in a rectangular lattice

    Parameters
    ----------
    dimX : int
        Number of pixels in the x direction.
    dimY : int
        Number of pixels in the y direction.
    outletPos : int
        Position of the outlet pixel.
    outletSide : int
        Side of the outlet pixel.
    cellSize : float
        Size of the pixels.
    seed : int, optional
        Seed for the random number generator. The default is 42.
    slope : float, optional
        Slope of the landscape. The default is 0.01.
    undirected : bool, optional
        Whether the OCN is returned as an undirected graph
        or not. The default is False.
    return_adj : bool, optional
        Whether the adjacency matrix of the OCN is returned
        or not, as a numpy array. The default is False.
        If the adjacency matrix is returned, the igraph object
        is not.

    Returns
    -------
    OCN : igraph object
        OCN as an R-igraph object.
    E : numpy array
        Energy dissipated by the OCN along the optimization process.
    areas : numpy array
        Accumulated area per each pixel in the OCN.
    Zmat : numpy array
        Elevation of each pixel in the OCN.
    OCNlandscape : igraph object
        OCN object
    """
    base.set_seed(int(seed))
    OCN = OCNet.create_OCN(dimX = dimX, dimY = dimY,
                           outletPos = outletPos, outletSide = outletSide,
                           saveEnergy = True, cellsize = cellSize,
                           displayUpdates = 0, initialNoCoolingPhase = 0.05)
    
    OCNlandscape = OCNet.landscape_OCN(OCN, slope = slope)

    Zvec = np.array(dollar(dollar(OCNlandscape, "FD"), "Z"))
    Zmat = np.reshape(Zvec, (dimX, dimY))

    OCNgraph = OCNet.OCN_to_igraph(OCNet.aggregate_OCN(OCNlandscape), level = "FD")

    if undirected:
        OCNgraph = igraphR.as_undirected(OCNgraph, mode = "collapse")

    E = np.array(dollar(OCN, "energy"))
    areas = np.array(dollar(dollar(OCN, "FD"), "A"))/cellSize**2

    if return_adj:
        OCNgraph = get_adjacency_R(OCNgraph)

    return OCNgraph, E, areas, Zmat, OCNlandscape

def get_aggregated_OCN(OCNlandscape, thrA):
    """
    Aggregates an OCN after thresholding the areas.

    Parameters
    ----------
    OCNlandscape : igraph object
        OCN as an R-igraph object.
    thrA : float
        Threshold for the areas.

    Returns
    -------
    adj : numpy array
        Adjacency matrix of the aggregated OCN.
    pos : numpy array
        Positions of the nodes in the aggregated OCN.
    AG_mask : numpy array
        Mask of the aggregated OCN.
    """

    dimX = dollar(OCNlandscape, "dimX")[0]
    dimY = dollar(OCNlandscape, "dimY")[0]

    OCN_AG = OCNet.aggregate_OCN(OCNlandscape, thrA = thrA)
    RN_idx = np.array(dollar(dollar(OCN_AG, "FD"), "toRN"))

    AG_RN_idx = np.array(dollar(dollar(OCN_AG, "RN"), "toAG"))
    AG_mask = np.zeros(dimX*dimY)

    counter = 0
    for i in range(dimX*dimY):
        if RN_idx[i] != 0:
            if AG_RN_idx[counter] != 0:
                AG_mask[i] = 1
            counter += 1

    AG_mask = np.reshape(AG_mask, (dimX, dimY))

    nNodes_Ag = dollar(dollar(OCN_AG, "AG"), "nNodes")[0]

    pos = np.zeros((nNodes_Ag, 2))

    for i in range(nNodes_Ag):
        pos[i, 0] = dollar(dollar(OCN_AG, "AG"), "X")[i]
        pos[i, 1] = dollar(dollar(OCN_AG, "AG"), "Y")[i]

    adj = get_adjacency_R(OCNet.OCN_to_igraph(OCN_AG, level = "AG"))
    adj = (adj + adj.T)/2

    return adj, pos, AG_mask



def get_adjacency_R(igraph_G):
    """
    Returns the adjacency matrix of an igraph object.

    Parameters
    ----------
    igraph_G : igraph object
        Graph to be converted.

    Returns
    -------
    adj : numpy array
        Adjacency matrix of the graph.
    """
    adj = np.array(base.as_matrix(igraphR.as_adjacency_matrix(igraph_G)))

    return adj

def find_metapop_OCN(OCNgraph, alpha):
    """
    Computes the metapopulation capacity of an OCN, using
    Hanski's model.

    Parameters
    ----------
    OCNgraph : igraph object
        OCN as an R-igraph object.
    alpha : float
        Characteristic dispersal length.

    Returns
    -------
    lambdaM : float
        Metapopulation capacity of the OCN.
    """

    dmat = np.array(igraphR.distances(OCNgraph))
    K = np.exp(-dmat/alpha)

    return utils.find_max_eigval(K)

def find_metapop_OCN_fixdim(dimX, dimY, nRep, alpha = 10, seeds = None,
                        cellSize = 500, slope = 0.01,
                        verbose = False):
    """
    Computes the metapopulation capacity and the energy dissipates in an OCN
    with fixed dimensions, by generating random OCNs with different outlets.
    All OCNs have the same dimensions, cell size and slope, and they are all
    with a single outlet.

    Parameters
    ----------
    dimX : int
        Number of pixels in the x direction.
    dimY : int
        Number of pixels in the y direction.
    nRep : int
        Number of OCNs to be generated.
    alpha : float, optional
        Characteristic dispersal length. The default is 10.
    seeds : numpy array, optional
        Seeds for the random number generator. The default is None.
        If None, the seeds are generated deterministically.
    cellSize : float, optional
        Size of the pixels. The default is 500.
    slope : float, optional
        Slope of the landscape. The default is 0.01.
    verbose : bool, optional
        Whether to print the progress or not. The default is False.

    Returns
    -------
    lambdaM_array : numpy array
        Metapopulation capacity of each OCN.
    E_array : numpy array
        Energy dissipated by each OCN.
    """
    
    if seeds is None:
        seeds = np.arange(nRep)
    cardinal_pos = ["N", "S", "E", "W"]

    lambdaM_array = np.zeros(nRep)
    E_array = np.zeros(nRep)

    for i in range(nRep):
        if verbose:
            print("Running OCN number: ", i)
        outletSide = np.random.choice(4)

        if outletSide == 1 or outletSide == 2:
            outletPos = np.random.choice(dimX) + 1
        else:
            outletPos = np.random.choice(dimY) + 1
        
        if verbose:
            print("\t Outlet size:", cardinal_pos[outletSide])
            print("\t Outlet position:", outletPos)

        OCNgraph, E, _ = createOCN(dimX, dimY,
                                   outletPos = outletPos,
                                   outletSide = cardinal_pos[outletSide],
                                   cellSize = cellSize,
                                   seed = int(seeds[i]), slope = slope,
                                   undirected = True)
        E_array[i] = E[-1]
        lambdaM_array[i] = find_metapop_OCN(OCNgraph, alpha)

        if verbose:
            print("\t Metapopulation capacity:", lambdaM_array[i])
            print("\t Energy dissipated:", E_array[i])

    return lambdaM_array, E_array
        
def generate_Pareto_OCNs(NRep, dims_array, alpha, cellSize = 500, slope = 0.01,
                         savePath = None, verbose = False):
    """
    Generates a set of OCNs with different dimensions, and computes the
    metapopulation capacity and the energy dissipated by each OCN.

    Parameters
    ----------
    NRep : int
        Number of OCNs to be generated per dimension.
    dims_array : numpy array
        Array of dimensions for the OCNs.
    alpha : float
        Characteristic dispersal length.
    cellSize : float, optional
        Size of the pixels. The default is 500.
    slope : float, optional
        Slope of the landscape. The default is 0.01.

    Returns
    -------
    lambdaM_array : numpy array
        Metapopulation capacity of each OCN.
    E_array : numpy array
        Energy dissipated by each OCN.
    """

    lambdaM_array = np.zeros((NRep, len(dims_array)))
    E_array = np.zeros((NRep, len(dims_array)))

    for idx, dims in enumerate(dims_array):
        if verbose:
            print("Running OCNs with dimensions:", dims)
            t0 = measure_time.time()
        seeds = np.random.randint(0, int(1e8), NRep)
        lambdaM_array[:, idx], E_array[:, idx] = find_metapop_OCN_fixdim(int(dims[0]), int(dims[1]), NRep, alpha = alpha,
                                                                     cellSize = cellSize, slope = slope,
                                                                     seeds = seeds, verbose = False)
        if verbose:
            print("Elapsed time:", measure_time.time() - t0)
            print("Mean metapopulation capacity:", lambdaM_array[:, idx].mean())
            print("Mean energy dissipated:", E_array[:, idx].mean())
        if savePath is not None:
            s = savePath + str(dims[0]) + "_" + str(dims[1])
            np.save(s + "_lambdaM.npy", lambdaM_array[:,idx])
            np.save(s + "_energy.npy", E_array[:,idx])

    return lambdaM_array, E_array