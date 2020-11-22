import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood
from getLikelihood import getLikelihood

def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####
    skin_data = estGaussMixEM(sdata, K, n_iter, epsilon)
    nons_data = estGaussMixEM(ndata, K, n_iter, epsilon)
    result = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rbg = img[i, j]
            pskin = getLikelihood(skin_data[1], skin_data[0], skin_data[2], rbg)
            pnonskin = getLikelihood(nons_data[1], nons_data[0], nons_data[2], rbg)
            judge = pskin/pnonskin
            if judge > theta:
                result[i,j] = 1
            else:
                result[i,j] = 0

    return result
