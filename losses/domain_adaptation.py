import torch
import torch.nn as nn


def norm(x, p: int):
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
    dist = norm(diff, 2)  # [Nx,Ny]
    return dist


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
    beta = 1. / (2. * torch.DoubleTensor(sigma * x.shape[0]).unsqueeze(1)).to(x.device)  # [Ns,1]
    dist = pairwise_distance_torch(x, y)  # [Nx,Ny]
    s = beta.float() @ dist.reshape(1, -1).float()  # [Ns,1]*[1,Nx*Ny]=[Ns,Nx*Ny]
    s = torch.exp(-s)
    return s.sum(axis=0).reshape(dist.shape)  # [Nx,Ny]


def F_mmd_loss(source, target, sigma):
    k = gaussian_kernel_torch
    nx, ny = source.shape[0], target.shape[0]
    cxx, cyy, cxy = 1 / (nx * (nx - 1)), 1 / (ny * (ny - 1)), 2 / (nx * ny)
    kxx, kyy, kxy = k(source, source, [sigma]), k(target, target, [sigma]), k(source, target, [sigma])
    mmd2 = cxx * (kxx.sum() - kxx.trace()) + cyy * (kyy.sum() - kyy.trace()) - cxy * kxy.sum()
    return mmd2


class MMDLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super(MMDLoss, self).__init__()
        self.sigma = sigma

        return

    def forward(self, source, target):
        return F_mmd_loss(source, target, self.sigma)


def l2diff(x1, x2):
    """
    standard euclidean norm. small number added to increase numerical stability.
    """
    return torch.sqrt(torch.sum((x1 - x2)**2))


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = (sx1 ** k).mean(0)
    ss2 = (sx2 ** k).mean(0)
    return l2diff(ss1, ss2)


def F_cmd_loss(x1, x2, n_moments):
    mx1 = x1.mean(dim=0)
    mx2 = x2.mean(dim=0)

    centered_x1 = x1 - mx1
    centered_x2 = x2 - mx2

    first_moment_diff = l2diff(mx1, mx2)

    if not torch.isinf(first_moment_diff):
        moments_diff_sum = first_moment_diff
    else:
        moments_diff_sum = 0
        print('Nan or Inf in loss...')

    for i in range(n_moments - 1):
        # moment diff of centralized samples
        diff_moments = moment_diff(centered_x1, centered_x2, i+2)

        if not (torch.isinf(diff_moments) or torch.isnan(diff_moments)):
            moments_diff_sum = moments_diff_sum + diff_moments
        else:
            print('Nan or Inf in loss...')

    return moments_diff_sum


class CMDLoss(nn.Module):
    def __init__(self, n_moments=3):
        super(CMDLoss, self).__init__()
        self.n_moments = n_moments

    def forward(self, x1, x2):
        return F_cmd_loss(x1, x2, self.n_moments)
