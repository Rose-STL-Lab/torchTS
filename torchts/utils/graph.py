import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


def normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    I = sp.eye(adj.shape[0])
    L = I - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return L


def random_walk(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def reverse_random_walk(adj_mx):
    return random_walk(np.transpose(adj_mx))


def scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if sp.issparse(adj_mx):
        adj_mx = adj_mx.todense()

    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    L = normalized_laplacian(adj_mx)

    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]

    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format="csr", dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L


def sparse_matrix(L):
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
    indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
    L = torch.sparse_coo_tensor(indices.T, L.data, L.shape)
    return L
