from concurrent.futures.process import _MAX_WINDOWS_WORKERS
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import matrix_power, solve
import warnings


'''
mu          -   Mean of random vector
P           -   Covariance matrix
x_skew      -   Vector of diagonal components of the skewness tensor
x_kurt      -   Vector of diagonal components of the kurtosis tensor
lb          -   Vector of lower bound of the state
ub          -   Vector of upper bound of the state

OUTPUTS

x           -   Matrix of sigma points
weights     -   Vector of weights corresponding to each sigma point
s           -   Vector of proportionality constants used to generate
                the sigma points
'''
def generalized_ut(mu, P, x_skew=None, x_kurt=None, lb=None, ub=None):

    # Get the number of states
    n = len(mu)
    # Evaluate the matrix square root via singular value decomposition
    # U1, S, Vh = np.linalg.svd(P)
    # C = U1 * np.diag(np.sqrt(np.diag(S))) * U1
    C = sqrtm(P)
    # Handle the arguments for skewness and kurtosis
    if x_skew == None:    # If no diagonal component of skewness is specified
        warnings.warn('No skewness specified: Gaussian skewness and kurtosis is assumed')
        x_skew = 0*mu          # Assume gaussian skewness if not provided
        x_kurt = 3*np.ones(n)   # Assume gaussian diagonal kurtosis in turn
    
    if x_kurt == None:    # If no diagonal component of kurtosis is specified
        warnings.warn('No kurtosis specified: kurtosis is selected to satisfy skewness kurtosis relationship')
        # Manually ensure kurtosis skew relationship is satisfied
        x_kurt = matrix_power(C, 4) * matrix_power(solve((matrix_power(C, 3)), x_skew), 2)      
        x_kurt = 1.1 * x_kurt
    
    # Handle when specified kurtosis violates skewness kurtosis relationship
    minkurt = matrix_power(C, 4) * matrix_power(solve((matrix_power(C, 3)), x_skew), 2)

    if (np.sum(x_kurt < minkurt)):
        warnings.warn('Bad Human Error: Kurtosis does not correspond to a distribution')
        for i in range(len(x_kurt)):
            if x_kurt[i] < minkurt[i]:
                x_kurt[i] = 1.001* minkurt[i]
        
    # Handle the arguments for lower bounds and upper bounds
    if lb == None: # If lower bound is not specified manually set lower bound as -inf
        lb = -np.inf * np.ones(n)

    if ub == None: # If lower bound is not specified manually set lower bound as -inf
        ub = np.inf * np.ones(n)

    # Calculate parameters u and v
    u = 0.5* ( -(solve(matrix_power(C, 3), x_skew) ) 
        + np.sqrt(4 * solve(matrix_power(C, 4), x_kurt) 
        - 3 * matrix_power(solve(matrix_power(C, 3) , x_skew), 2)))
    v = u + solve(matrix_power(C, 3), x_skew)
    # Generate the sigma points
    x0 = mu
    x1 = mu - C * np.diag(u)
    x2 = mu + C * np.diag(v)

    # # --------------- This section handles the constraints  --------------- 
    # Flag_constrain = 0     # Default flag to enforce constraint
    # # Check if mean violates constraints
    # if np.subtract(mu, lb).min() < 0 or np.subtract(mu, lb).min() < 0:
    #     Flag_constrain = 1  # Set flag to avoid enforcing state constraints
    #     warnings.warn('Unable fo enforce constraints: one or more of the mean does not satisfy lb < mean < ub')

    # if Flag_constrain == 0:
    #     theta = 0.9;    # Default value of user defined slack parameter
        
    #     # Ensure lower bound 'lb' is not violated
    #     Temp1 = np.subtract(np.hstack((x1, x2)), lb)
    #     L1 = np.nonzero(Temp1.min() < 0)   # Find the location of sigma points that violate the lower bound
    #     Flag_calc = 0;      # Flag that determines if skewness can be matched
    #     print(u)
    #     print(v)
    #     print(n)
    #     print(L1)
    #     for i in range(len(L1)):
    #         if L1[i] <= n:
    #             # Recalculate 'u' to satisfy lower bound 'lb'
    #             u[L1[i]] = theta * np.min(np.abs(np.subtract(mu, lb) / C[:, L1[i]]))
    #         else:
    #             # Recalculate 'v' to satisfy lower bound 'lb'
    #             print(L1[i])
    #             v[L1[i] - n] = theta * np.min(np.abs(np.subtract(mu, lb) / C[:, L1[i]-n]))
    #             Flag_calc = 1   # Set flag

    #     # Regenerate the sigma points
    #     x1 = mu - C * np.diag(u)
    #     x2 = mu + C * np.diag(v)
        
    #     #     Ensure upper bound 'ub' is not violated
    #     Temp2 = ub  - np.hstack((x1, x2))
    #     L2 = np.nonzero(min(Temp2) < 0)    # Find the location of sigma points that
    #                                        # violate the upper bound
    #     for i in range(len(L2)): 
    #         if L2(i) <= n:
    #             # Recalculate 'u' to satisfy upper bound 'ub'
    #             u[L2[i]] = theta * np.min(np.abs(np.subtract(mu, lb) / C[:, L1[i]]))
    #         else:
    #             # Recalculate 'v' to satisfy upper bound 'ub'
    #             v[L2[i] - v] = theta * np.min(np.abs(np.subtract(ub, mu) / C[:, L2[i]-n]))
    #             Flag_calc = 1   # Set flag

    #     if Flag_calc == 0:
    #         # Now recalculate parameter 'v' to match diagonal componen of 
    #         # skewness tensor because it was 'v' was not previously redefined
    #         v = u + np.solve(matrix_power(C, 3), x_skew)  # only done of v was not redefined
        
    #     # Regenerate the sigma points to reflect any change in 'u' or 'v'
    #     x1 = mu - C * np.diag(u)
    #     x2 = mu + C * np.diag(v)

    # Recalculate weights to reflect any change in 'u' or 'v'
    w2 = np.ones(n) / v  / np.add(u, v)
    w1 = w2 * v / u
    # Output sigma point values
    x = np.hstack((x0, x1, x2))
    w0 = 1 - np.sum(np.vstack((w1, w2)), axis = 0)
    weights = np.transpose(np.hstack((w0, np.transpose(w1), np.transpose(w2))))
    # s = np.vstack((u, v))
    # print(x)
    # print(weights)
    # print(s)
    return x, weights