import numpy as np
def getLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood
    #####Insert your code here for subtask 6a#####

    num_dims = X.shape[0]
    num_gauss = len(weights)
    likelihood = 0
    sum_weighted = 0
    for k in range(num_gauss):
        cov_matrix_k = covariances[:, :, k]
        sum_weighted += weights[k]*1 / ((2 * np.pi) ** (num_dims / 2) * (np.linalg.det(cov_matrix_k)) ** 0.5) * np.exp(
            -0.5 * (X - means[k]).dot(np.linalg.inv(cov_matrix_k)).dot(X - means[k]))
    likelihood = sum_weighted

    return likelihood