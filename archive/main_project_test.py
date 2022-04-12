import numpy as np
import math
import matplotlib.pyplot as plt
from extended_kalman_filter_project import extended_kalman_filter
import json

class myStruct:
    pass
    
    
def process_model(x):
    """
    Think this makes sense because we don't move at all
    :param x:
    :return:
    """
    f = np.array([x[0,0], x[1,0], x[2,0]], dtype=float)
    return f.reshape([3, 1])

def wraptopi(x):
    # wraps angles in x, in radians, to the interval [−pi, pi] such that pi maps to pi and −pi maps to −pi. 
    # In general, odd, positive multiples of pi map to pi and odd, negative multiples of pi map to −pi.
    pi = np.pi
    x = x - np.floor(x/(2*pi)) *2 *pi
    print('x', x)
    if x>=pi:
        x = x-2*pi
    else:
        x = x
    return float(x)

def measurement(Z_landmark):
    # use this function to transform the [x, y, z] 3D coordinate got from pixel position measurement to the 2D bearing and range measurement
    # mx, my is the x, y coordinate of the image 1 (the image at time t/t+1), it is the coordinate of the landmark.
    # landmarks at t are only used for getting the bearing_range measurement.
    
    Z_landmark = Z_landmark.reshape((3,1))
    Z_2D = Z_landmark[0:2, 0]
    mx = Z_2D[0]
    print('mx', mx)
    my = Z_2D[1]
    z_br = np.array([[math.atan2(my, mx)],[np.sqrt(my ** 2 + mx ** 2)]])  # z_bearing_range
    return z_br.reshape([2, 1])

def measurement_model(x_hat, mx_t_minus_1, my_t_minus_1):
    # hfun
    # mx_t_minus_1 is the x coordinate of the image 1 at time t-1, it is the coordinate of the landmark.
    # x_hat is the estimated state by the process_model 
    """
    :param mx_t_minus_1, my_t_minus_1: the x, y coordinate of the image 1 (the image at time t-1), it is the coordinate of the landmark.
    :param x_hat: the estimated state by the process_model, [x, y ,theta]
    :return: h
    """
    x_hat = x_hat.reshape((-1,1))
    mx = mx_t_minus_1
    my = my_t_minus_1
    h = np.array([[wraptopi(math.atan2(my - x_hat[1,0], mx - x_hat[0,0])) - x_hat[2,0]], [np.sqrt((my - x_hat[1,0]) ** 2 + (mx - x_hat[0,0]) ** 2)]])

    # sys.hfun = @(landmark_x, landmark_y, mu_pred) [...
    # wrapToPi(atan2(landmark_y - mu_pred(2), landmark_x - mu_pred(1)) - mu_pred(3));
    # sqrt((landmark_y - mu_pred(2))^2 + (landmark_x - mu_pred(1))^2)];
    temp = math.atan2(my - x_hat[1,0], mx - x_hat[0,0])
    print('hfun', temp)
    print('wrap', wraptopi(temp))

    return h.reshape([2, 1])

def measurment_Jacobain(landmark_x, landmark_y, mu_pred, z_hat):
    # Hfun
    """
    :param landmark_x, landmark_y: the x, y coordinate of the image 1 (the image at time t-1), it is the coordinate of the landmark. matrix
    :param mu_pred: the estimated state by the process_model, [x, y ,theta]. matrix
    :param z_hat: hfun. matrix
    :return: mesaurment jacobian
    """
    z_hat = z_hat.reshape((-1,))
    mu_pred = mu_pred.reshape((-1,))

    Hfun = np.array([[(landmark_y - mu_pred[1])/(z_hat[1]**2), -(landmark_x - mu_pred[0])/(z_hat[1]**2), -1],
                     [-(landmark_x - mu_pred[0])/z_hat[1],    -(landmark_y - mu_pred[1])/z_hat[1],      0]])
    # print('Hfun', Hfun)
    return Hfun.reshape((2,3))


if __name__=="__main__":

    # Opening JSON file
    f = open('dict.json')

    # returns JSON object as a dictionary
    data = json.load(f)
    xyz_t = data['Frame_t+1']
    xyz_t_1 = data['Frame_t']


    f.close()

    z_1 = np.zeros((len(xyz_t), 2))
    z_2 = np.zeros((len(xyz_t), 2))
    z_3 = np.zeros((len(xyz_t), 2))
    landmark1 = np.zeros((len(xyz_t), 2))
    landmark2 = np.zeros((len(xyz_t), 2))
    landmark3 = np.zeros((len(xyz_t), 2))
    for i in range(len(xyz_t)):
        z_xyz_1 = np.array([xyz_t[i][0][10], xyz_t[i][1][10], xyz_t[i][2][10]])
        z_xyz_2 = np.array([xyz_t[i][0][-15], xyz_t[i][1][-15], xyz_t[i][2][-15]])
        z_xyz_3 = np.array([xyz_t[i][0][0], xyz_t[i][1][0], xyz_t[i][2][0]])
        # t-1
        z_xyz_1_1 = np.array([xyz_t_1[i][0][10], xyz_t_1[i][1][10], xyz_t_1[i][2][10]])
        z_xyz_2_1 = np.array([xyz_t_1[i][0][-15], xyz_t_1[i][1][-15], xyz_t_1[i][2][-15]])
        z_xyz_3_1 = np.array([xyz_t_1[i][0][0], xyz_t_1[i][1][0], xyz_t_1[i][2][0]])

        landmark1[i,:] = z_xyz_1_1[0:2]
        landmark2[i,:] = z_xyz_2_1[0:2]
        landmark3[i,:] = z_xyz_3_1[0:2]

        z_1_i = measurement(z_xyz_1)
        z_2_i = measurement(z_xyz_2)
        z_3_i = measurement(z_xyz_3)
        # print('z1i',type(z_1_i[0,0]))
        z_1[i, :] = z_1_i.reshape((2,))
        z_2[i, :] = z_2_i.reshape((2,))
        z_3[i, :] = z_3_i.reshape((2,))
    
    t = list(range(len(xyz_t)))
    t = [float(x) for  x in t]

    # initialize the state using the first measurement
    init = myStruct()
    init.x1 = np.zeros([3,1])
    # init.x1[0,0] = 46.6
    # init.x1[1,0] = -33.7
    # init.x1[2, 0] = -1.95
    init.x1[0,0] = 0
    init.x1[1,0] = 0
    init.x1[2, 0] = 0

    init.Sigma = np.eye(3) #Tune


    # Build the system
    sys = myStruct()
    sys.A = np.eye(3)  # double check size
    sys.B = []  # no input because the object is static
    sys.f = process_model
    sys.H = measurment_Jacobain # 2 by3
    # sys.H2 = measurment_Jacobain_2 # # 2by 3
    sys.R1 = np.cov(z_1, rowvar=False)  # double check 2by 2
    sys.R2 = np.cov(z_2, rowvar=False)  # double check 2 by 2
    sys.R3 = np.cov(z_3, rowvar=False)  # double check 2 by 2
    sys.h = measurement_model

    # sys.Q1 = 0.01*np.array([[0.03, 0.02, 0.01],
    #                    [0.02, 0.04, 0.01],
    #                    [0.01, 0.01, 0.05]]).reshape((3, 3))  # vibrations M
    
    sys.Q1 = 10*np.eye(3) # vibrations M
    # sys.Q1 = 50*np.array([[0.03, 0.02, 0.01],
    #                    [0.02, 0.03, 0.02],
    #                    [0.02, 0.01, 0.01]]).reshape((3, 3))  # vibrations M good

    # sys.Q1 = 10*np.array([[0.03, 0.02, 0.015],
    #                         [0.02, 0.3, 0.015],
    #                         [0.015, 0.015, 0.1]]).reshape((3, 3))  # vibrations M 

    sys.Q1 = 1*np.array([[0.03, 0.02, 0.015],
                        [0.02, 0.3, 0.015],
                        [0.015, 0.015, 0.1]]).reshape((3, 3))  # vibrations M 
    
    # sys.Q1 = np.zeros((3,3))


    # ################################################
    # implement ekf
    ekf_1b = extended_kalman_filter(sys, init)
    x1_b = []
    x1_b.append(init.x1)

    # started from 0 instead cuz picked random init
    for i in range(0, np.shape(z_1)[0],1):
        # print('typez1',type(z_1[0,0]))
        ekf_1b.prediction_1()
        zstack = np.append(z_1[i, :], z_2[i, :]).reshape((-1,1))
        zstack = np.append(zstack, z_3[i, :]).reshape((-1,1))
        print('zstack',zstack)
        # print(zstack)

        ekf_1b.correction_batch(zstack, landmark1[i, :], landmark2[i, :],landmark3[i, :])
        # since the ekf_1b.x1 is the relative location to the x0, we add it to the x0 location.
        # double check for the covariance.
        x1_b.append(x1_b[-1] + ekf_1b.x1)

    # change the axes and add the initilization point.
    x1_b = np.array(x1_b).reshape((-1,3))
    x1_b_1 = np.zeros(x1_b.shape)
    x1_b_1[:,0] = -x1_b[:,1]
    x1_b_1[:,1] = x1_b[:,0]
    # x1_b = x1_b_1
    # x1_b[:,0] = -x1_b[:,0]
    true_initialization = np.array([46.6, -33.7, -1.95])
    # true_initialization = np.array([85, 15, -1.95])
    x1_b_1 = x1_b_1 + true_initialization
    x1_b_1 = x1_b_1.reshape((-1,3))


    # Final Label
    print('Final x: %.4f, y: %.4f, z: %.4f' % (x1_b[-1,0], x1_b[-1,1], x1_b[-1,2]))
    # plotting
    fig = plt.figure()
    plt.plot(x1_b_1[:, 0], x1_b_1[:, 1])
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    # plt.legend()
    plt.title('EKF Batch Measurement')
    plt.show()
