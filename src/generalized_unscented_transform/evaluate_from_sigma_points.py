from numpy.linalg import matrix_power
import numpy as np



'''
 This function is a tool that is used to evaluate the sample statistics 
 of sigma points of an unscented transform

 INPUTS

   sigma_points        -   Matrix of sigma points
   weights             -   Vector of weights corresponding to each sigma point

 OUTPUTS

   sigma_mean          -   Sample mean of sigma points
   sigma_cov           -   Sample covariance matrix of sigma points
   sigma_skew          -   Sample diagonal component of skewness tensor
   sigma_kurt          -   Sample diagonal component of kurtosis tensor
'''

def Evaluate_sample_statistics(sigma_points, weights):

    n = sigma_points.shape[0]
    # row_weights = []
    # weights = weights.reshape(1, -1) # Convert to row vector (need testing)
    # Mean
    sigma_mean = np.sum(np.matmul(sigma_points, np.tile(weights,(n, 1))))
    # Covariance
    Z = (sigma_points - sigma_mean) 
    # print(np.diag(np.transpose(weights)[:,:]))
    sigma_cov = np.sum(Z * np.diag(np.array(np.transpose(weights))[0])* np.transpose(Z))
    # Diagonal skewness
    sigma_skew = np.sum(np.matmul(np.power(Z, 3), np.tile(weights,(n, 1))))
    # Diagonal kurtosis
    sigma_kurt = np.sum(np.matmul(np.power(Z, 4), np.tile(weights,(n, 1))))
    return sigma_mean, sigma_cov, sigma_skew, sigma_kurt
