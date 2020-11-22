import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).
    n,k = np.shape(gamma)

    D = np.shape(X[1])
    buf = np.zeros(D)
    N_dach = np.zeros(k)
    weights = np.zeros(k)
    d = D[0]
    'k = K[0]'
    'n = N[0]'
    means = np.zeros((k, d))
    covariances = np.zeros((d, d, k))

    for i in range(k):
        N_dach[i] = sum(gamma[:, i])
        weights[i] = N_dach[i]/n
        for m in range(n):
            means[i, :] += gamma[m, i] * X[m, :]/N_dach[i]
        for q in range(n):
            temp = X[q, :] - means[i, :]
            temp1 = temp.reshape(temp.shape[0], 1)
            temp2 = temp1.dot(temp1.transpose())
            covariances[:, :, i] += gamma[q, i] / N_dach[i] * temp2

    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    #####Insert your code here for subtask 6c#####
    return weights, means, covariances, logLikelihood
