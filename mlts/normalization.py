import numpy as np

def estimate(X, axis = 0):
    """Estimate normalization parameters of the specified features."""

    mu = np.mean(X, axis=axis)
    sigma = np.std(X, ddof=1, axis=axis)
    return mu, sigma

def apply(X, mu, sigma):
    """Apply normalization to the specified features."""

    ## If all elements the same 'sigma[i] == 0', don't apply normilization
    sigma[sigma == 0] = 1
    return (X - mu) / sigma
