import numpy as np
def getLogLikelihood(means, weights, covariances, X):
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

    num_datapoints = X.shape[0]
    num_dims = X.shape[1]
    num_gauss = len(weights)
    logLikelihood = 0
    for n in range(num_datapoints):
        sum_weighted_nk = 0
        for k in range(num_gauss):
            cov_matrix_k = covariances[:, :, k]
            sum_weighted_nk += weights[k]*1 / ((2 * np.pi) ** (num_dims / 2) * (np.linalg.det(cov_matrix_k)) ** 0.5) * np.exp(
                -0.5 * (X[n, :] - means[k]).dot(np.linalg.inv(cov_matrix_k)).dot(X[n, :] - means[k]))
        log_sum_weighted_nk = np.log(sum_weighted_nk)
        logLikelihood += log_sum_weighted_nk
    return logLikelihood