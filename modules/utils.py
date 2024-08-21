import numpy as np
import networkx as nx
import scipy.optimize as optimize
import matplotlib

import copy
import model

from numba import njit, prange
import os



def exp_fit(x, characteristic, amplitude):
    """
    Exponential function.

    Parameters
    ----------
    x : numpy.ndarray
        Array of x values.
    characteristic : float
        Characteristic decay length.
    amplitude : float
        Amplitude of the exponential.

    Returns
    -------
    y : numpy.ndarray
        Array of y values.
    """
    return amplitude*np.exp(-x/characteristic)


####################
# NETWORK ANALYSIS #
####################

def get_average_degree(net):
    """
    Finds the average degree of a network.

    Parameters
    ----------
    net : nx.Graph
        Network.

    Returns
    -------
    average_degree : float
        Average degree of the network.
    """
    average_degree = np.mean([d for _, d in net.degree()])

    return average_degree


@njit(parallel = True)
def find_laplacian_nb(adj_matrix):
    """
    Finds the Laplacian matrix of a network.

    Parameters
    ----------
    adj_matrix : numpy.ndarray
        Adjacency matrix of the network.

    Returns
    -------
    laplacian : numpy.ndarray
        Laplacian matrix of the network.
    """
    out_deg = np.sum(adj_matrix, axis=0)
    N = adj_matrix.shape[0]

    laplacian = np.zeros((N,N), dtype = np.float64)

    for i in range(N):
        for j in range(N):
            laplacian[i,j] = (i==j)*out_deg[i] - adj_matrix[i,j]

    return laplacian

def find_laplacian(network):
    """
    Finds the Laplacian matrix of a network.

    Parameters
    ----------
    network : nx.Graph
        Network.

    Returns
    -------
    laplacian : numpy.ndarray
        Laplacian matrix of the network.
    """
    adj_matrix = nx.adjacency_matrix(network).toarray()

    return find_laplacian_nb(adj_matrix)

@njit
def DFSUtil(A, clist, node, visited):
    '''
    Depth-first search.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of the network.
    clist : List
        List of nodes in the current cluster.
    node : int
        Current node.
    visited : numpy.ndarray
        Array of visited nodes.

    Returns
    -------
    clist : List
        List of nodes in the current cluster.
    '''
    N = A.shape[0]
    
    # Mark the current vertex as visited
    visited[node] = True
 
    # Store the vertex to list
    clist.append(node)
 
    # Get nearest neighbours
    nn = []
    nn.append(0)
    nn.remove(0)
    for n in range(N):
        if (not node == n) and A[node, n]>0:
            nn.append(n)
                
    # Repeat for all nn
    for i in nn:
        if visited[i] == False:
            # Update the list
            clist = DFSUtil(A, clist, i, visited)

    return clist

@njit
def find_connected_components(A):
    '''
    Method to retrieve connected components
    in an undirected graph.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of the network.

    Returns
    -------
    cc : List
        List of nodes in each connected components.
    '''
    N = A.shape[0]

    visited = np.zeros(N, dtype=np.bool_)
    cc = []
    
    # Loop over nodes
    for v in range(N):
        if visited[v] == False:
            # if not visited, compute cluster
            clist = []
            clist.append(0)
            clist.remove(0)
            
            clust = DFSUtil(A, clist, v, visited)
            cc.append(clust)
    
    return cc

def fragment_all(network, edges_to_remove):
    """
    Fragments a network by removing given edges.

    Parameters
    ----------
    network : nx.Graph
        Network to be fragmented.
    edges_to_remove : list
        List of edges to be removed.
        
    Returns
    -------
    frag_network : nx.Graph
        Fragmented network.
    """

    frag_network = copy.deepcopy(network)

    for edge in edges_to_remove:
        frag_network.remove_edge(*edge)

    return frag_network

def fragment_communities(network, edges_to_remove, N_modules, Nodes_per_module):
    """
    Fragments a network by sequentially removing its communities.

    Parameters
    ----------
    network : nx.Graph
        Network to be fragmented.
    edges_to_remove : list
        List of edges between the different communities.
    N_modules : int
        Number of comunities.
    Nodes_per_module : list
        Number of nodes in each community.

    Returns
    -------
    fragmented_networks : list
        List of networks, each with one less community connected.
    """
    fragmented_networks = []

    cumsum = np.concatenate([[0], np.cumsum(Nodes_per_module)])

    for i in range(N_modules-1):
        if i == 0:
            net = copy.deepcopy(network)
        else:
            net = copy.deepcopy(fragmented_networks[i-1])
        for edge in edges_to_remove:
            if (edge[0] < cumsum[i+1] and edge[0] >= cumsum[i]) or (edge[1] < cumsum[i+1] and edge[1] >= cumsum[i]):
                if net.has_edge(*edge):
                    net.remove_edge(*edge)
        fragmented_networks.append(net)

        if nx.number_connected_components(net) == N_modules:
            break

    return fragmented_networks


def comm_removal(net_generator, N_modules, Nodes_per_module, kwargs,
                 f, xi, seed = None):
    """
    Computes the metapopulation capacity of a modular network by fragmenting it
    and computing the metapopulation capacity of each fragment.

    Parameters
    ----------
    net_generator : function
        Function that generates the network.
    N_modules : int
        Number of comunities.
    Nodes_per_module : list
        Number of nodes in each community.
    kwargs : dict
        Keyword arguments for the network generator.
    f : float
        Fraction of nodes removed.
    xi : float
        Fraction of edges removed.
    seed : int, optional
        Seed for the random number generator. The default is None.
        If None, the seed is randomly generated.

    Returns
    -------
    lambdaMax : float
        Metapopulation capacity of the network.
    """

    if seed is None:
        seed = np.random.randint(0, int(1e9))
    
    modular_network, edges_modular = net_generator(N = Nodes_per_module, Nmod = N_modules,
                                                   seed = seed, **kwargs)

    fragmented_networks = fragment_communities(modular_network, edges_modular, N_modules, Nodes_per_module)

    network_list = [modular_network, *fragmented_networks]

    adj_list = [nx.to_numpy_array(net) for net in network_list]

    lambdaMax = model.metapop_capacity_fragmented(adj_list, f, xi)

    return lambdaMax, network_list

def average_comm_removal(net_generator, N_modules, Nodes_per_module, kwargs,
                         f, xi, Nrep = 10, seeds = None):
    """
    Average over multiple realizations of a modular network by fragmenting it
    and computing the average metapopulation capacity at each fragmentation step.

    Parameters
    ----------
    net_generator : function
        Function to generate the modular network.
    N_modules : int
        Number of comunities.
    Nodes_per_module : list
        Number of nodes in each community.
    kwargs : dict
        Keyword arguments for the network generator.
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
    Nrep : int, optional
        Number of realizations to average over. The default is 10.
    seeds : list, optional
        List of seeds for the random number generator. The default is None.
        If None, the seeds are randomly generated.

    Returns
    -------
    lambdaMax_all : numpy.ndarray
        Array of metapopulation capacities at each fragmentation step.
    """
    if seeds is None:
        seeds = np.random.randint(0, int(1e9), Nrep)
    
    lambdaMax_all = []

    for idx_rep in range(Nrep):

        print(f'Rep {idx_rep+1}/{Nrep}')

        lambdaMax, _ = comm_removal(net_generator, N_modules, Nodes_per_module,
                                    kwargs, f, xi, seed = seeds[idx_rep])
        
        lambdaMax_all.append(lambdaMax)

    return np.array(lambdaMax_all)


####################
# HELPER FUNCTIONS #
####################

def euclidean_distance_matrix(points, return_unique = False):
    """
    Computes the euclidean distance matrix between a set of points.

    Parameters
    ----------
    points : numpy.ndarray
        Set of points.
    return_unique : bool, optional
        If True, returns the unique values of the distance matrix. The default
        is False.

    Returns
    -------
    dist_mat : numpy.ndarray
        Euclidean distance matrix.
    unique_dist : numpy.ndarray
        Unique values of the distance matrix.
        Only returned if return_unique is True.
    """
    dist_mat = np.sqrt(np.sum((points[:, None, :] - points[None, :, :])**2, axis = -1))
    # dist_mat = np.sum(abs(points[:, None, :] - points[None, :, :]), axis = -1)
    if return_unique:
        return dist_mat, np.unique(dist_mat)
    else:
        return dist_mat

def euclidean_distance_matrix_1D(points):
    """
    Computes the euclidean distance matrix between a set of points.

    Parameters
    ----------
    points : numpy.ndarray
        Set of points.

    Returns
    -------
    dist_mat : numpy.ndarray
        Euclidean distance matrix.
    """
    dist_mat = np.sqrt((points[:, None] - points[None, :])**2)
    return dist_mat

@njit
def off_diag(mat):
    """
    Returns all the off-diagonal elements of a matrix.

    Parameters
    ----------
    A : np.array
        Matrix.

    Returns
    -------
    np.array
        Off-diagonal elements of A.
    """
    N = mat.shape[0]
    vals = []
    for i in range(N):
        for j in range(N):
            if i != j:
                vals.append(mat[i,j])
                
    return np.array(vals)

@njit
def diagonalize_matrix_sym(mat):
    """
    Numba wrapper for diagonalizing a symmetric matrix.

    Parameters
    ----------
    mat : numpy.ndarray
        Matrix to be diagonalized.

    Returns
    -------
    eigvals : numpy.ndarray
        Eigenvalues of the matrix.
    eigvecs : numpy.ndarray
        Eigenvectors of the matrix.
    eigvecs_inv : numpy.ndarray
        Inverse of the eigenvectors of the matrix.

    Returns
    -------
    eigvals : numpy.ndarray
        Eigenvalues of the matrix.
    eigvecs : numpy.ndarray
        Eigenvectors of the matrix.
    eigvecs_inv : numpy.ndarray
        Inverse of the eigenvectors of the matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvecs_inv = np.linalg.inv(eigvecs)

    return eigvals, eigvecs, eigvecs_inv


@njit
def diagonalize_matrix(mat):
    """
    Numba wrapper for diagonalizing a non symmetric matrix.

    Parameters
    ----------
    mat : numpy.ndarray
        Matrix to be diagonalized.

    Returns
    -------
    eigvals : numpy.ndarray
        Eigenvalues of the matrix.
    eigvecs : numpy.ndarray
        Eigenvectors of the matrix.
    eigvecs_inv : numpy.ndarray
        Inverse of the eigenvectors of the matrix.

    Returns
    -------
    eigvals : numpy.ndarray
        Eigenvalues of the matrix.
    eigvecs : numpy.ndarray
        Eigenvectors of the matrix.
    eigvecs_inv : numpy.ndarray
        Inverse of the eigenvectors of the matrix.
    """

    eigvals, eigvecs = np.linalg.eig(mat)
    eigvecs_inv = np.linalg.inv(eigvecs)

    return eigvals, eigvecs, eigvecs_inv


@njit
def create_diag_matrix(array):
    """
    Creates a diagonal matrix from an array.

    Parameters
    ----------
    array : numpy.ndarray
        Array to be converted to a diagonal matrix.

    Returns
    -------
    diag_matrix : numpy.ndarray
        Diagonal matrix.
    """
    N = array.size

    diag_matrix = np.zeros((N,N), dtype = type(array[0]))

    for i in range(N):
        for j in range(N):
            if i == j:
                diag_matrix[i,j] = array[i]

    return diag_matrix

@njit
def extract_submatrix(mat, idx_list):
    """
    Extracts a submatrix from a matrix using numba.

    Parameters
    ----------
    mat : numpy.ndarray
        Matrix from which to extract the submatrix.
    idx_list : list
        List of indices of the rows and columns to be extracted.

    Returns
    -------
    submat : numpy.ndarray
        Submatrix.
    """
    N = mat.shape[0]

    idx_list = np.array(idx_list, dtype = np.int64)

    submat = np.zeros((len(idx_list), len(idx_list)), dtype = mat.dtype)

    for i in range(len(idx_list)):
        for j in range(len(idx_list)):
            submat[i,j] = mat[idx_list[i], idx_list[j]]
    
    return submat

def get_files(path):
    """
    Get the list of files in a directory.

    Parameters
    ----------
    path : str
        Path to the directory.
    
    Returns
    -------
    files : list
        List of files in the directory.
    """
    files = []

    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(file)

    return files

@njit
def linear_to_tuple(linear_index, shape):
    """
    Convert a linear index to a tuple index.

    Parameters
    ----------
    linear_index : int
        Linear index.
    shape : tuple
        Shape of the array.

    Returns
    -------
    i : int
        Row index.
    j : int
        Column index.
    """
    i = linear_index // shape[1]
    j = linear_index % shape[1]
    return i, j

@njit
def tuple_to_linear(i, j, shape):
    """
    Convert a tuple index to a linear index.

    Parameters
    ----------
    i : int
        Row index.
    j : int
        Column index.
    shape : tuple
        Shape of the array.

    Returns
    -------
    linear_index : int  
        Linear index.
    """
    return i*shape[1] + j

@njit
def sigmoid_inv(x, beta = 1):
    """
    Returns a sigmoid function starting from 1 and decreasing to 0.

    Parameters
    ----------
    x : numpy array
        Input array.
    beta : float, optional
        Slope of the sigmoid function.

    Returns
    
    """
    return 1-1/(1+np.exp(-beta*x))

@njit
def exponential_range(x, xmin, xmax, ymin, ymax, beta = 1):
    """
    Decreasing exponential function of x that is equal to ymax at xmin and
    ymin at xmax.

    Parameters
    ----------
    x : numpy array
        Input array.
    xmin : float
        Minimum value of x.
    xmax : float
        Maximum value of x.
    ymin : float
        Minimum value of the output.
    ymax : float
        Maximum value of the output.
    beta : float, optional
        Slope of the exponential function.
    
    Returns
    -------
    y : numpy array
        Output array.
    """
    a = ymax + np.exp(beta * xmax)*(ymax - ymin) / (np.exp(beta * xmin) - np.exp(beta * xmax))
    b = (ymax - ymin) * np.exp(beta*(xmax + xmin)) / (- np.exp(beta * xmin) + np.exp(beta * xmax))

    return a + b*np.exp(-beta*x)

def load_DEM_model_data(path, name, load_kernel = False):
    f_array = np.load(path + name + '_f_array.npy')

    lambdaMax = []
    if load_kernel:
        K_array = []
    rho_array = []

    for i, f in enumerate(f_array):
        lambdaMax.append(np.load(path + name + f'_lambdaM_f{f}.npy'))
        if load_kernel:
            K_array.append(np.load(path + name + f'_K_f{f}.npy'))
        rho_array.append(np.load(path + name + f'_rhostat_f{f}.npy'))

    if load_kernel:
        K_array = np.array(K_array)
    lambdaMax = np.array(lambdaMax)
    rho_array = np.array(rho_array)

    if load_kernel:
        return f_array, K_array, lambdaMax, rho_array
    else:
        return f_array, lambdaMax, rho_array
    
def load_DEM_Hanski_data(path, name):
    alpha_array = np.load(path + name + '_Hanski_alpha_array.npy')

    rho_array = []
    for i, f in enumerate(alpha_array):
        rho_array.append(np.load(path + name + f'_Hanski_rhostat_alpha{f}.npy'))

    rho_array = np.array(rho_array)

    return alpha_array, rho_array


######################
# NETWORK GENERATORS #
######################

def generate_net_degree(generator, target_degree, target_error,
                        kwargs={}, select_return = None, verbose = False):
    """
    Generates a network with a given degree distribution.

    Parameters
    ----------
    generator : function
        Function that generates a network.
    target_degree : float
        Target average degree of the network.
    target_error : float
        Target error in the average degree of the network.
    kwargs : dict
        Keyword arguments for the generator function.
    select_return : int
        Index of the return of the generator function to be used.
    verbose : bool
        Whether to print the average degree of the network.

    Returns
    -------
    net : nx.Graph
        N
    """
    average_degree = 0
    while np.abs(average_degree - target_degree) > target_error:
        res = generator(**kwargs)
        if select_return is None:
            net = res
        else:
            net = res[select_return]
        average_degree = get_average_degree(net)
        if verbose:
            print(average_degree)

    return res

def generate_connected_ER(N, p):
    """
    Generates a connected Erdos-Renyi network.

    Parameters
    ----------
    N : int
        Number of nodes in the network.
    p : float
        Probability of connection between nodes.

    Returns
    -------
    connected_network : nx.Graph
        Connected Erdos-Renyi network.
    """
    connected_network = nx.erdos_renyi_graph(N, p)
    while not nx.is_connected(connected_network):
        connected_network = nx.erdos_renyi_graph(N, p)

    return connected_network

def generate_connected_scale_free(N, alpha = 0.41, beta = 0.54, gamma = 0.05, seed = None):
    """
    Generates a connected scale-free network.

    Parameters
    ----------
    N : int
        Number of nodes in the network.
    alpha : float
        Probability of adding a new node connected to an existing node chosen
        randomly according to the in-degree distribution.
    beta : float
        Probability of adding an edge between two existing nodes. One existing
        node is chosen randomly according the in-degree distribution and the
        other chosen randomly according to the out-degree distribution.
    gamma : float
        Probability of adding a new node connected to an existing node chosen
        randomly according to the out-degree distribution.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    connected_network : nx.Graph
        Connected scale-free network.
        Self-loops are removed, and the network is converted to undirected, so
        the original relation between exponent and degree distribution is lost.
    """
    connected_network = nx.scale_free_graph(N, alpha = alpha, beta = beta, gamma = gamma,
                                            seed = seed)
    connected_network = connected_network.to_undirected()

    connected_network = nx.Graph(connected_network)

    connected_network.remove_edges_from(nx.selfloop_edges(connected_network))


    while not nx.is_connected(connected_network):
        connected_network = nx.scale_free_graph(N, alpha = alpha, beta = beta, gamma = gamma)
        connected_network = connected_network.to_undirected()
        connected_network.remove_edges_from(nx.selfloop_edges(connected_network))

    return connected_network

def generate_configuration_model_powerlaw(N, exp, seed = None, scale = 1.75,
                                          target_degree = None, error = 0.001):
    """
    Generates a configuration model network with a power-law degree distribution.
    Note that the final network may not be connected.

    Parameters
    ----------
    N : int
        Number of nodes in the network.
    exp : float
        Exponent of the power-law degree distribution.
    seed : int
        Seed for the random number generator.
    target_degree : float
        Target average degree of the network.
    error : float
        Target error in the average degree of the network.

    Returns
    -------
    G : nx.Graph
        Configuration model network.
    """
    degree_sequence = np.random.zipf(exp, int(N*scale))
    if target_degree is not None:
        while abs(np.mean(degree_sequence) - target_degree) > error:
            degree_sequence = np.random.zipf(exp, int(N*scale))

    G = nx.configuration_model(degree_sequence, seed = seed)
    G.remove_edges_from(nx.selfloop_edges(G))
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    while len(G) != N:
        G = nx.configuration_model(degree_sequence, seed = seed)
        G.remove_edges_from(nx.selfloop_edges(G))
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(Gcc[0])

    return G

@njit
def generate_arbitrary_scale_free(N, L, exponent=3.0, directed=False):
    """
    Generates scale-free graphs of given size and exponent.

    It follows the method proposed by Goh, Kahng & Kim 'Universal Behaviour
    of Load Distribution in SF networks' PRL 87 (2001). Every node is chosen
    with probability alpha = 1/(gamma - 1) where gamma is the desired exponent.

    Note that the average degree is given by 2*L/N.

    Parameters
    ----------
    N : integer
        Number of nodes of the final network.
    L : integer
        Number of links of the resulting random network.
    exponent : float (optional)
        The exponent (in positive) of the degree-distribution of the resulting
        networks. Recommended values between 2 and 3.
    directed : boolean (optional)
        False if a graph is desired, True if digraphs are desired. In case
        of digraphs, both the input and the output degrees follow a scale-
        free distribution but uncorrelated between them.

    Returns
    -------
    adj : ndarray of rank-2 and size NxN.
        The adjacency matrix of the generated scale-free network.

    Notes
    -----
    In case of directed networks the input and the output degrees of the
    nodes are correlated, e.g., input hubs are also output hubs.
    """
    maxL = 0.5*N*(N-1)
    if directed:
        maxL= N*(N-1)
        if L > maxL:
            print("L out of bounds, max(L) = N*(N-1) =", maxL)
    else:
        if L > maxL:
            print( "L out of bounds, max(L) = 1/2*N*(N-1) =", maxL)

    adj = np.zeros((N,N), np.uint8)

    # Create a degree sequence
    alpha = 1.0/(exponent - 1.0)
    nodeweights = (np.arange(N) +1)**-alpha
    # Probability of a node to be chosen
    nodeweights /= nodeweights.sum()
    nodecumprobability = nodeweights.cumsum()

    counter = 1
    while counter <= L:

        # 2.1) Choose two nodes to connect
        xa = np.random.rand()
        xasum = np.sum(np.sign(nodecumprobability-xa))
        a = int(0.5*(N-xasum))

        xb = np.random.rand()
        xbsum = np.sum(np.sign(nodecumprobability-xb))
        b = int(0.5*(N-xbsum))

        # 2.2) Do not allow self loops and multiple edges
        if a == b: continue
        if adj[a,b]: continue

        adj[a,b] = 1
        if not directed:
            adj[b,a] = 1
        counter += 1

    return adj

def generate_modular_ER(N, Nmod, p_intra, p_connect, Nconn_per_node = 1, seed=0):
    """
    Generates a modular Erdos-Renyi network.

    Parameters
    ----------
    N : list
        Number of nodes in each module.
    Nmod : int
        Number of modules in the network.
    p_intra : float or list
        Probability of connection within a module.
    p_connect : float
        Probability of connection between modules.
    Nconn_per_node : int
        Number of connections per node that connect the
        different modules.
    seed : int
        Seed for random number generator.

    Returns
    -------
    modular_network : networkx.Graph
        Modular Erdos-Renyi network.
    connecting_edges : list
        List of edges that connect the different modules.
    """
    np.random.seed(seed)

    modular_network = nx.Graph()

    if isinstance(p_intra, float):
        p_intra = [p_intra]*Nmod

    for i in range(Nmod):
        ER_net = generate_connected_ER(N[i], p_intra[i])
        # relabel nodes
        ER_net = nx.relabel_nodes(ER_net, lambda x: x + i*N[i])
        # add to modular network
        modular_network = nx.disjoint_union(modular_network, ER_net)
    
    # add inter-module connections
    nodes_to_connect = []

    for i in range(Nmod):
        nodes_to_connect.append(np.random.choice(range(i*N[i], (i+1)*N[i]), size=int(p_connect*N[i]), replace=False))

    connecting_edges = []

    for i in range(Nmod):
        for node in nodes_to_connect[i]:
            modules_to_connect = np.random.choice(list(set(range(Nmod)) - set([i])), 
                                                  size=Nconn_per_node, replace = False)
            for j in modules_to_connect:
                to_add = False

                while not to_add:
                    node_j = np.random.choice(nodes_to_connect[j])
                    to_add = not modular_network.has_edge(node, node_j)
                    to_add = to_add and not modular_network.has_edge(node_j, node)
                modular_network.add_edge(node, node_j)
                connecting_edges.append((node, node_j))      

    return modular_network, connecting_edges


def generate_corridor_ER(N, Nmod, p_intra, p_connect, Nconn_per_node = 1, seed=0,
                         periodic = False):
    """
    Generates an ecological-corridor Erdos-Renyi network, where different communities
    are connected sequentially.

    Parameters
    ----------
    N : list
        Number of nodes in each module.
    Nmod : int
        Number of modules in the network.
    p_intra : float or list
        Probability of connection within a module.
    p_connect : float
        Probability of connection between modules.
    Nconn_per_node : int
        Number of connections per node that connect the
        different modules.
    seed : int
        Seed for random number generator.

    Returns
    -------
    modular_network : networkx.Graph
        Modular Erdos-Renyi network.
    connecting_edges : list
        List of edges that connect the different modules.
    """
    np.random.seed(seed)

    modular_network = nx.Graph()

    if isinstance(p_intra, float):
        p_intra = [p_intra]*Nmod

    for i in range(Nmod):
        ER_net = generate_connected_ER(N[i], p_intra[i])
        # relabel nodes
        ER_net = nx.relabel_nodes(ER_net, lambda x: x + i*N[i])
        # add to modular network
        modular_network = nx.disjoint_union(modular_network, ER_net)
    
    # add inter-module connections
    nodes_to_connect = []

    for i in range(Nmod):
        nodes_to_connect.append(np.random.choice(range(i*N[i], (i+1)*N[i]), size=int(p_connect*N[i]), replace=False))

    connecting_edges = []

    for i in range(Nmod - 1):
        j = i + 1
        for _ in range(Nconn_per_node):
            for node in nodes_to_connect[i]:
                to_add = False
                while not to_add:
                    node_j = np.random.choice(nodes_to_connect[j])
                    to_add = not modular_network.has_edge(node, node_j)
                    to_add = to_add and not modular_network.has_edge(node_j, node)
                modular_network.add_edge(node, node_j)
                connecting_edges.append((node, node_j))

    if periodic:
        i = Nmod - 1
        j = 0
        for _ in range(Nconn_per_node):
            for node in nodes_to_connect[i]:
                to_add = False
                while not to_add:
                    node_j = np.random.choice(nodes_to_connect[j])
                    to_add = not modular_network.has_edge(node, node_j)
                    to_add = to_add and not modular_network.has_edge(node_j, node)
                modular_network.add_edge(node, node_j)
                connecting_edges.append((node, node_j))

    return modular_network, connecting_edges

######################
# PLOTTING FUNCTIONS #
######################

import matplotlib.pyplot as plt

def plot_modular(net, colors_list, Nodes_per_module, pos = None, ax = None,
                 node_size = 40, alpha_nodes = 1, lw_nodes = 0.5, ec_nodes = 'k',
                 lw_edges = 0.5, alpha_edges = 0.5, ec_edges = 'k',):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    ax.axis('off')

    cumsum = np.concatenate([[0], np.cumsum(Nodes_per_module)])
    node_colors = []
    for i in range(len(Nodes_per_module)):
        node_colors += [colors_list[i]]*Nodes_per_module[i]

    if pos is None:
        N_modules = len(Nodes_per_module)

        centers = np.array([[np.cos(2*np.pi*i/N_modules), np.sin(2*np.pi*i/N_modules)] for i in range(N_modules)])
        new_positions = np.zeros((len(net), 2))

        for i in range(N_modules):
            x = np.random.normal(loc = centers[i, 0], scale = 0.1, size = Nodes_per_module[i])
            y = np.random.normal(loc = centers[i, 1], scale = 0.1, size = Nodes_per_module[i])

            new_positions[cumsum[i]:cumsum[i+1], 0] = x
            new_positions[cumsum[i]:cumsum[i+1], 1] = y
        
        pos = {i: (new_positions[i,0], new_positions[i,1]) for i in range(len(net))}

    nx.draw_networkx_nodes(net, pos = pos, node_size=node_size,
                           alpha=alpha_nodes, linewidths=lw_nodes, edgecolors=ec_nodes,
                           node_color = node_colors, ax = ax)
    nx.draw_networkx_edges(net, pos = pos, width=lw_edges, alpha=alpha_edges,
                           edge_color=ec_edges, ax = ax)

    if ax is None:
        return fig, ax, pos, node_colors
    else:
        return pos, node_colors
    
import matplotlib.colors as colors

def white_to_color_map(cmap_color):
    """
    Generates a colormap that goes from white to a given color.

    Parameters
    ----------
    cmap_color : str
        Color to which the colormap goes.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colormap that goes from white to the given color.
    """
    cmap = colors.LinearSegmentedColormap.from_list("", ["white", cmap_color])
    return cmap

def colors_to_color_map(cmap_colors, nodes = None):
    """
    Generates a colormap from a list of colors.

    Parameters
    ----------
    cmap_colors: list
        List of colors for the colormap.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colormap.
    """
    if nodes is None:
        nodes = np.linspace(0,1, len(cmap_colors))
    cmap = colors.LinearSegmentedColormap.from_list("", list(zip(nodes, cmap_colors)))
    return cmap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=1000):
    """
    Truncate a colormap between two values.

    Parameters
    ----------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colormap to be truncated.
    minval : float
        Minimum value of the colormap.
    maxval : float
        Maximum value of the colormap.
    n : int
        Number of values in the truncated colormap.

    Returns
    -------
    new_cmap : matplotlib.colors.LinearSegmentedColormap
        Truncated colormap.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap

def darken_cmap(cmap, scale = 0.5):
    """
    Darken a colormap.

    Parameters
    ----------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colormap to be darkened.
        If a string is passed, the corresponding matplotlib colormap is used.
    scale : float
        Scale factor for darkening.

    Returns
    -------
    new_cmap : matplotlib.colors.ListedColormap
        Darkened colormap.
    """

    if type(cmap) == str:
        cmap = matplotlib.colormaps[cmap]
    cmap = cmap(np.arange(cmap.N))
    cmap[:, 0:3] *= scale

    return matplotlib.colors.ListedColormap(cmap)

def plot_with_lines(x, y, color, s, lw, label, ax, ec = 'white', lw_ec = 0.1, alpha = 0.9,
                    marker = 'o'):
    ax.scatter(x, y, s = s, color = color, label = label, ec = ec, lw = lw_ec,
               zorder = np.inf, marker = marker)
    ax.plot(x, y, color = color, lw = lw, alpha = alpha)


def plot_kernel_distances(distances, avg_K, min_K, max_K, color, label, alpha_fill = 0.1,
                          lw_avg = 3, s = 55, lw_ec = 1, alpha_avg = 0.8, lw_dash = 0.5,
                          alpha_dash = 0.2, ax = None, marker = 'o'):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    plot_with_lines(distances, avg_K, color = color, s = s, lw_ec = lw_ec,
                    lw = lw_avg, label = label, ax = ax, alpha = alpha_avg, marker = marker)
    ax.fill_between(distances, min_K, max_K, color = color, alpha = alpha_fill, lw = 0)
    ax.plot(distances, min_K, color = color, alpha = alpha_dash, lw = lw_dash, ls = '--')
    ax.plot(distances, max_K, color = color, alpha = alpha_dash, lw = lw_dash, ls = '--')

    ax.set_yscale('log')
    ax.set_xlabel(r'Network distance')
    ax.set_ylabel(r'Average kernel')

    return ax



def find_xi_rescaling(network, f, target_lambdaM, check=True):
    """
    network: nx network object
    f : exploration parameter    
    target_lambdaM : desired metapopulation capacity
    check: bool, if true computes and returns the new metapop. capacity
    Returns:
    new_xi : float, the xi needed to have the desired metapop. capacity
    new_lambda : float, the new metapop. capacity or None if check=False
    """
    kernel = model.find_effective_kernel(f, 1.0, network)
    current_lambdaM = model.find_metapopulation_capacity(kernel, network, f, 1.0)
    new_xi = target_lambdaM / current_lambdaM
    if check:
        new_lambda = model.find_metapopulation_capacity(kernel, network, f, new_xi)
    else:
        new_lambda = None

    return new_xi, new_lambda
