"""
Taken from: https://github.com/chamwen/DaNN_DJP/blob/master/djp_mmd.py
"""
import torch
from utils import DEVICE


# -------------------------------------------------------------------
def _rbf_kernel(Xs, Xt, sigma):
    Z = torch.cat((Xs, Xt), 0)
    ZZT = torch.mm(Z, Z.T)
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.T
    K = torch.exp(-exponent / (2 * sigma ** 2))
    return K


# -------------------------------------------------------------------
# functions to compute the marginal MMD with rbf kernel
def rbf_mmd(Xs, Xt, sigma):
    ones = lambda nb: torch.ones(nb, 1).to(DEVICE)
    K = _rbf_kernel(Xs, Xt, sigma)
    m = Xs.size(0)  # assume Xs, Xt are same shape
    e = torch.cat((1 / m * ones(m), -1 / m * ones(m)), 0)
    M = e * e.T
    tmp = torch.mm(torch.mm(K, M), K.T)
    loss = torch.trace(tmp)
    return loss
