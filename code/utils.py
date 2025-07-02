import numpy as np
import celerite2

# this functions were made to compute the LOO log predictive probability for a Gaussian process regression model
# for the case 

# manually compute the LOO log probability density

def stabilize_matrix(a, jitter=1e-6):
    """
    Add a small jitter to the diagonal of a matrix to stabilize it.
    """
    a[np.arange(a.shape[0]), np.arange(a.shape[1])] += jitter
    # a = a + jitter
    return a
def add_diagonal(a, diag):
    """
    Add a diagonal to a matrix.
    """
    a[np.arange(a.shape[0]), np.arange(a.shape[1])] += diag
    return a

def lag_matrix(y, x):  # y is the one with one missing value
    """
    Compute the lag matrix: the difference between `y` and `x`.
    """
    matrix = y[:, None] - x[None, :]
    return matrix

def sho_term(sigma, rho, Q=1.0 / 3):
    """Create a SHOTerm kernel with given parameters."""
    return celerite2.terms.SHOTerm(sigma=sigma, rho=rho, Q=Q)

def rotation_term(sigma, period, Q0, dQ, f):
    """Create a Rotation term kernel with given parameters."""
    return celerite2.terms.RotationTerm(sigma=sigma, period=period, Q0=Q0, dQ=dQ, f=f)

def logp(indice, x, y, diag, sigma, term=None, rho=None, period=None, Q0=None, dQ=None, f=None, verbose=False):
    """Compute the LOO log predictive probability for a single point."""
    # instanciate the kernel
    if term == 'sho':
        kernel = sho_term(sigma, rho)
    elif term == 'rotation':
        kernel = rotation_term(sigma, period, Q0, dQ, f)
    else:
        raise ValueError("term must be 'sho' or 'rotation'")

    # compute the intermediate matrices
    # m1 = kernel.get_value(lag_matrix(np.array([x[indice]]), np.delete(x,indice)))  # m1 has small values, makes sense for pairs of points too separated, ~1e-6
    # m2 = kernel.get_value(lag_matrix(np.delete(x,indice), np.delete(x,indice)))
    # m2 = add_diagonal(m2, np.delete(diag, indice))
    # m3 = kernel.get_value(lag_matrix(np.array([x[indice]]), np.array([x[indice]])))    # the input to get_value here is always 0, because there's only one test point in  K(x*, x*)
    # m4 = kernel.get_value(lag_matrix(np.delete(x,indice), np.array([x[indice]])))

    # compute the mean and covariance of the predictive distribution for the left-out point
    # mean = np.dot(m1, np.linalg.solve(m2, np.delete(y, indice)))
    # cov = m3 - np.dot(m1, np.linalg.solve(m2, m4))

    # alternative way
    m5 = kernel.get_value(lag_matrix(x, x))
    m5 = add_diagonal(m5, diag)

    # mean = y[indice] - np.linalg.solve(m5, y)[indice] / np.linalg.solve(m5, np.identity(len(x)))[indice, indice]
    # cov = 1 / np.linalg.solve(m5, np.identity(len(x)))[indice, indice]

    m5_inv = np.linalg.solve(m5, np.identity(len(x)))
    mean = y[indice] - np.dot(m5_inv, y)[indice] / m5_inv[indice, indice]
    cov = 1 / m5_inv[indice, indice]

    # compute the log predictive probability
    term1 = -0.5*np.log(cov)
    term2 = -0.5*(y[indice]-mean)**2/cov 
    term3 = -0.5*np.log(2*np.pi)
    
    if verbose:
        print('indice:', indice, 'cov:', cov, 'mean:', mean, "term1:", term1, "term2:", term2, "term3:", term3, "diag:", diag[indice])
    return term1 + term2 + term3

def loo_cv(x, y, diag, sigma, term=None, rho=None, period=None, Q0=None, dQ=None, f=None, verbose=False):
    """Compute the full LOO-CV log predictive probability.
    term (str): 'sho' or 'rotation'
    """
    logp_values = [logp(i, x, y, diag, sigma, term=term, rho=rho, period=period, Q0=Q0, dQ=dQ, f=f, verbose=verbose) for i in range(len(x))]

    loo = np.sum(logp_values)
    loo_se = (len(x)*np.var(logp_values))**0.5
    return loo, loo_se

def log_likelihood(x, y, diag, term, sigma, rho=None, period=None, Q0=None, dQ=None, f=None):
    """log-Likelihood function for N data points yn at points tn
    with variance diag and kernel term.
    this does the same as the compute_log_likelihood function in pymc"""
    if term == 'sho':
        kernel = sho_term(sigma, rho)
    elif term == 'rotation':
        kernel = rotation_term(sigma, period, Q0, dQ, f)
    else:
        raise ValueError("term must be 'sho' or 'rotation'")
    
    m1 = kernel.get_value(lag_matrix(x, x))
    m1 = add_diagonal(m1, diag)

    term1 = -0.5*np.dot(y.T, np.linalg.solve(m1, y))
    term2 = -0.5*np.log(np.linalg.det(m1))
    term3 = -0.5*len(x)*np.log(2*np.pi)

    return term1 + term2 + term3
