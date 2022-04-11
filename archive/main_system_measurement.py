import numpy as np
from numpy.random import randn
from particle_filter import particle_filter
import matplotlib.pyplot as plt
import math



class myStruct:
    pass

def projection_matrix_left(fx, fy, cx, cy):
    '''return the projection_matrix_left

    Parameters
    ----------
    fx: float
        fx = f/k, where k is the pixel density, pixels/m, f is the focus length
        Similarly, fy = f/l

    Returns
    -------
    3*4 matrix
        projection_matrix_left

    '''
    P_left = np.array([[fx, 0, cx, 0],[0, fy, cy, 0],[0, 0, 1, 0]])
    return P_left

def projection_matrix_right(fx, fy, cx, cy, R, T):
    '''return the projection_matrix_right

    Parameters
    ----------
    fx: float
        fx = f/k, where k is the pixel density, pixels/m, f is the focus length
        Similarly, fy = f/l

    Returns
    -------
    3*4 matrix
        projection_matrix_right

    '''
    # construct the trasformation matrix, which transforms poses from the frame of camera1 to that of camera2
    T = T.reshape((3,1))
    RT = np.vstack((R, T))
    last_row = np.array([0,0,0,1])
    RT = np.hstack((RT, last_row))
    # transform the points from the frame of camera1 to that of camera2, so we can build the Pr
    P_right = np.array([[fx, 0, cx, 0],[0, fy, cy, 0],[0, 0, 1, 0]])
    P_right = P_right @ RT

    return P_right

def linear_triangulation(x_left, x_right, P_left, P_right):
    '''return the 3D and 2D coordinate of the point in the left camera frame

    Parameters
    ----------
    x_left, x_right: pixel coordinate in the left image
        can either be 2*1 or 3*1 homogeneous form
    P_left, P_right: projection matrices of the left and right cameras

    Returns
    -------
    3*1 matrix
        Z: the 3D coordinate of the point in the left camera frame
    2*1 matrix
        Z_2D: the 2D projection coordinate

    '''
    x_left = x_left.reshape((-1,1))
    x_right = x_right.reshape((-1,1))
    # check if they are in homogeneous form, if not, then transform them
    if x_left.shape == (3,1):
        pass
    else:
        x_left = np.array([x_left[0,0], x_left[1,0], 1]).reshape((3,1))

    if x_right.shape == (3,1):
        pass
    else:
        x_right = np.array([x_right[0,0], x_right[1,0], 1]).reshape((3,1))


    A = np.array([[x_left[0,:]*P_left[2,:]-P_left[0,:]],
                  [x_left[1,:]*P_left[2,:]-P_left[1,:]],
                  [x_right[0,:]*P_right[2,:]-P_right[0,:]],
                  [x_right[1,:]*P_right[2,:]-P_right[1,:]]])

    U, S_vector, V_transpose = np.linalg.svd(A, full_matrices=True)
    V = V_transpose.T

    X = V[:,3].reshape((-1,1))
    X = X * (1/X[3,:])
    X = X[0:3,:].reshape((3,1))
    Z_2D = np.array([X[2,:], X[0,:]]).reshape((2,1))  # (z,x) = (mx, my), check
    Z = X
    return Z, Z_2D

def measurement(x_left, x_right, P_left, P_right):
    # use this function to transform the pixel position measurement to the 2D bearing and range measurement
    Z, Z_2D = linear_triangulation(x_left, x_right, P_left, P_right)
    mx = Z_2D[0, 0]
    my = Z_2D[1, 0]
    z_br = np.array([[math.atan2(my, mx)],[np.sqrt(my ** 2 + mx ** 2)]])  # z_bearing_range
    return z_br.reshape([2, 1])

def measurement_model(x_hat, mx_t_minus_1, my_t_minus_1):
    x_hat = x_hat.reshape((-1,1))
    mx = mx_t_minus_1
    my = my_t_minus_1
    h = np.array([[math.atan2(my - x_hat[1,0], mx - x_hat[0,0]) - x_hat[2,0]], [np.sqrt((my - x_hat[1,0]) ** 2 + (mx - x_hat[0,0]) ** 2)]])
    return h.reshape([2, 1])




def process_model(x, w):
    """
    Think this makes sense because we don't move at all
    :param x: actually point with respect to frame 1
    :param w: noise vector sampled as part of the motion model (the vibes )
    :return:
    """
    f = np.array([x[0], x[1], x[2]], dtype=float).reshape([3,1]) + w
    return f.reshape([3, 1])

def measurement_model_1(K_f, p, c_1):
    """
    :param K_f:  intrinisic Camera 1 Matrix
    :param p: point in frame 1
    :param c_1: optical center of image camera 1
    :return: process model camera 1
    """
    proj_func= np.array([p[0]/p[2], p[1]/p[2]], dtype=float).reshape((-1,1))
    return np.dot(K_f, proj_func) + c_1

def measurement_model_2(K_f, p, c_2):
    """
    :param K_f: intrinisic Camera 2 Matrix
    :param p: point in frame 1
    :param c_1:
    :return:
    """
    p = p.reshape([3,1])
    q = np.dot(R.T, p) - np.dot(R.T, t)
    proj_func= np.array([q[0]/q[2], q[1]/q[2]], dtype=float).reshape((-1,1))
    return np.dot(K_f, proj_func) + c_2


if __name__=="__main__":

    C_1 = np.loadtxt(open('data/C_1.csv'), delimiter=",").reshape(-1, 1)
    C_2 = np.loadtxt(open('data/C_2.csv'), delimiter=",").reshape(-1, 1)
    Kf_1 = np.loadtxt(open('data/Kf_1.csv'), delimiter=",")
    Kf_2 = np.loadtxt(open('data/Kf_2.csv'), delimiter=",")
    R = np.loadtxt(open('data/R.csv'), delimiter=",")
    t = np.loadtxt(open('data/t.csv'), delimiter=",").reshape(-1, 1)
    z_1 = np.loadtxt(open('data/z_1.csv'), delimiter=",")
    z_2 = np.loadtxt(open('data/z_2.csv'), delimiter=",")

    given_data = myStruct()
    given_data.C_1 = C_1
    given_data.C_2 =C_2
    given_data.Kf_1 = Kf_1
    given_data.Kf_2 = Kf_2
    given_data.R = R
    given_data.t = t


    # initialize the state using the first measurement
    init = myStruct()
    init.x = np.zeros([3,1])
    init.x[0, 0] = 0.12
    init.x[1, 0] = 0.09
    init.x[2, 0] = 1.5
    init.n = 1000
    init.Sigma = 0.01*np.eye(3) #Tune


    # Build the system
    sys = myStruct()
    sys.f = process_model
    sys.R1 = np.cov(z_1, rowvar=False)  # double check 2by 2
    sys.R2 = np.cov(z_2, rowvar=False)  # double check 2 by 2
    sys.h1 = measurement_model_1
    sys.h2 = measurement_model_2
    sys.Q1 = np.array([[0.03,0.02,0.01],
                       [0.02,0.04, 0.01],
                       [0.01,0.01, 0.05]]).reshape((3,3)) # vibrations


    filter = particle_filter(sys, init, given_data)
    x = np.empty([3, np.shape(z_1)[0]+1]) # used since picked a random init
    x[:, 0] = [init.x[0, 0], init.x[1, 0], init.x[2, 0]]  # don't need since picked random point


    green = np.array([0.2980, .6, 0])

    # started from 0 instead cuz picked random init
    for i in range(0, np.shape(z_1)[0], 1):
        filter.sample_motion()

        #####################################################
        # First measurement
        z = np.append(z_1[i, :], z_2[i, :]).reshape((-1,1))

        filter.importance_mesurement_batch(z)
        if filter.Neff < filter.n/5:   # Note sampling threshold is hidden in there
            filter.resampling() # does this work correctly terms of shapes
        wtot = np.sum(filter.p.w)

        if wtot  > 0:
            # a = filter.p.x
            # b = filter.p.w
            x[0, i+1] = np.sum(filter.p.x[: , 0] * filter.p.w.reshape(-1)) / wtot
            x[1 ,i+1] = np.sum(filter.p.x[: , 1] * filter.p.w.reshape(-1))/ wtot
            x[2, i+1] = np.sum(filter.p.x[:, 2] * filter.p.w.reshape(-1)) / wtot
        else:
            print('\033[91mWarning: Total weight is zero or nan!\033[0m')
            x[:, i+1] = [np.nan, np.nan, np.nan]
        ############################################################################


    # Final Label
    print('Final x: %.4f, y: %.4f, z: %.4f' % (x[0,-1], x[1,-1], x[2,-1]))

    # quit()
    fig = plt.figure()
    # axis = np.linspace()
    plt.plot(x[0,:], label='px' )
    plt.plot(x[1,:], label='py')
    plt.plot(x[2,:], label ='pz')
    plt.title('Particle Filter Batch ')
    plt.xlabel(r'time Step')
    plt.ylabel(r'Position')
    plt.legend()
    plt.show()



    print('here')
