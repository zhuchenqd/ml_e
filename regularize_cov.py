import numpy as np


def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix

    #####Insert your code here for subtask 6d####
    a = covariance.shape[0]
    if covariance.ndim == 2:
        regularized_cov = covariance + epsilon * np.eye(a)
    else:
        for i in range(covariance.shape[2]):
            covariance[:, :, i] += epsilon * np.eye(a)
        regularized_cov = covariance
    return regularized_cov
