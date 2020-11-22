import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    # logLikelihood = 0
    num_dims = X.shape[1]
    num_gauss = len(means)
    num_datapoints = X.shape[0]
    gamma = np.zeros([num_datapoints, num_gauss])
    fenzi = 0
    logLikelihood = 0
    for j in range(num_gauss):
        for n in range(num_datapoints):
            sum_weighted_gauss = 0
            fenzi = weights[j]*1 / ((2 * np.pi) ** (num_dims / 2) * (np.linalg.det(covariances[:, :, j])) ** 0.5) * np.exp(
                -0.5 * (X[n, :] - means[j]).dot(np.linalg.inv(covariances[:, :, j])).dot(X[n, :] - means[j]))
            for k in range(num_gauss):
                '''denominator'''
                # cov_matrix_k = covariances[:, :, k]
                sum_weighted_gauss += weights[k]*1 / ((2 * np.pi) ** (num_dims / 2) * (np.linalg.det(covariances[:, :, k])) ** 0.5) * np.exp(
                -0.5 * (X[n, :] - means[k]).dot(np.linalg.inv(covariances[:, :, k])).dot(X[n, :] - means[k]))
            gamma[n, j] = fenzi/sum_weighted_gauss
    #print('gamma:', gamma)
    # return [logLikelihood, gamma]
    return [logLikelihood, gamma]