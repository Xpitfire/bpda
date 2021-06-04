import torch
import numpy as np


def norm_np(x, p: int):
    """
    Compute p-norm across the features of a batch of instances.
    Parameters
    ----------
    x
        Batch of instances of shape [N, features].
    p
        Power of the norm.
    Returns
    -------
    Array where p-norm is applied to the features.
    """
    return (x ** p).sum(axis=1)


def pairwise_distance_np(x, y):
    """
    Compute pairwise distance between 2 samples.
    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    Returns
    -------
    [Nx, Ny] matrix with pairwise distances.
    """
    assert len(x.shape) == len(y.shape) and len(x.shape) == 2 and x.shape[-1] == y.shape[-1]
    diff = np.reshape(x, x.shape + (1,)) - np.reshape(np.transpose(y),
                                                      (1,) + np.transpose(y).shape)  # [Nx,F,1]-[1,F,Ny]=[Nx,F,Ny]
    dist = norm_np(diff, 2)  # [Nx,Ny]
    return dist


def pairwise_distance_torch(x, y):
    """
    Compute pairwise distance between 2 samples.
    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    Returns
    -------
    [Nx, Ny] matrix with pairwise distances.
    """
    assert len(x.shape) == len(y.shape) and len(x.shape) == 2 and x.shape[-1] == y.shape[-1]
    diff = x.unsqueeze(2) - (y.T).unsqueeze(0)  # [Nx,F,1]-[1,F,Ny]=[Nx,F,Ny]
    dist = norm_np(diff, 2)  # [Nx,Ny]
    return dist


def gaussian_kernel_np(x, y, sigma):
    """
    Gaussian kernel between samples of x and y. A sum of kernels is computed
    for multiple values of sigma.
    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    sigma
        Array with values of the kernel width sigma.
    chunks
        Chunk sizes for x and y when using dask to compute the pairwise distances.
    Returns
    -------
    Array [Nx, Ny] of the kernel.
    """
    beta = 1. / (2. * np.expand_dims(sigma * x.shape[0], 1))  # [Ns,1]
    dist = pairwise_distance_np(x, y)  # [Nx,Ny]
    s = beta @ dist.reshape(1, -1)  # [Ns,1]*[1,Nx*Ny]=[Ns,Nx*Ny]
    s = np.exp(-s)  # if isinstance(s, np.ndarray) else da.exp(-s)
    return s.sum(axis=0).reshape(dist.shape)  # [Nx,Ny]


def gaussian_kernel_torch(x, y, sigma):
    """
    Gaussian kernel between samples of x and y. A sum of kernels is computed
    for multiple values of sigma.
    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    sigma
        Array with values of the kernel width sigma.
    chunks
        Chunk sizes for x and y when using dask to compute the pairwise distances.
    Returns
    -------
    Array [Nx, Ny] of the kernel.
    """
    beta = 1. / (2. * torch.DoubleTensor(sigma * x.shape[0]).unsqueeze(1))  # [Ns,1]
    dist = pairwise_distance_torch(x, y)  # [Nx,Ny]
    s = beta.double() @ dist.reshape(1, -1).double()  # [Ns,1]*[1,Nx*Ny]=[Ns,Nx*Ny]
    s = torch.exp(-s)
    return s.sum(axis=0).reshape(dist.shape)  # [Nx,Ny]


def mmd_np(x, y):
    """
    Compute maximum mean discrepancy between 2 samples.
    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    kernel
        Kernel function.
    sigma
        Bandwidth of Gaussian kernel.
    Returns
    -------
    MMD^2 between the samples x and y.
    """
    sigma = 0.1
    k = gaussian_kernel_np
    nx, ny = x.shape[0], y.shape[0]
    cxx, cyy, cxy = 1 / (nx * (nx - 1)), 1 / (ny * (ny - 1)), 2 / (nx * ny)
    kxx, kyy, kxy = k(x, x, [sigma]), k(y, y, [sigma]), k(x, y, [sigma])
    mmd2 = cxx * (kxx.sum() - kxx.trace()) + cyy * (kyy.sum() - kyy.trace()) - cxy * kxy.sum()
    return mmd2


def mmd_torch(x, y):
    """
    Compute maximum mean discrepancy between 2 samples.
    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    kernel
        Kernel function.
    sigma
        Bandwidth of Gaussian kernel.
    Returns
    -------
    MMD^2 between the samples x and y.
    """
    sigma = 0.1
    k = gaussian_kernel_torch
    nx, ny = x.shape[0], y.shape[0]
    cxx, cyy, cxy = 1 / (nx * (nx - 1)), 1 / (ny * (ny - 1)), 2 / (nx * ny)
    kxx, kyy, kxy = k(x, x, [sigma]), k(y, y, [sigma]), k(x, y, [sigma])
    mmd2 = cxx * (kxx.sum() - kxx.trace()) + cyy * (kyy.sum() - kyy.trace()) - cxy * kxy.sum()
    return mmd2


def cmd_np(x_s, x_t):
    """
    central moment discrepancy (cmd) in tensorflow

    Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning,"
    International Conference on Learning Representations, 2017.
    """
    # difference between means
    m_s = np.mean(x_s, 0)
    m_t = np.mean(x_t, 0)
    loss = np.linalg.norm(m_s - m_t)
    # difference between 2nd up to 3rd central moments
    for i in [2, 3]:
        m_s = np.mean(np.power(x_s, i), 0)
        m_t = np.mean(np.power(x_t, i), 0)
        loss = loss + np.linalg.norm(m_s - m_t)
    return loss
