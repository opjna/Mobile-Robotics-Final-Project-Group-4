import numpy as np
from scipy.linalg import block_diag

# Extended Kalman filter class for state estimation of a nonlinear system
class extended_kalman_filter:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        #
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.A = system.A  # system matrix Jacobian
        self.B = system.B  # input matrix Jacobian
        self.f = system.f  # process model
        self.H = system.H  # measurement model Jacobian
        # self.H2 = system.H2  # measurement model Jacobian
        self.R1 = system.R1  # measurement noise covariance
        self.R2 = system.R2  # measurement noise covariance
        self.h = system.h  # measurement model
        # self.h2 = system.h2  # measurement model
        self.x1 = init.x1  # state vector actually the mean

        # accounting for vibrations
        self.Q1 = system.Q1  # input noise covariance
        # self.Q2 = system.Q2  # input noise covariance

        self.Sigma = init.Sigma  # state covariance


        ########################3
        # Modified for Part b
        self.R_stack = block_diag(system.R1, system.R2,system.R3) #  prevents need to stack repeatedly


    def prediction_1(self):
        # good
        # EKF propagation (prediction) step
        # Taking care of the motion model in camera 1 here
        self.x1 = np.zeros((3,1))
        self.x_pred1 = self.f(self.x1)   # predicted state
        
        # Noise is addtive so W_k is idenity
        self.Sigma_pred1 = np.dot(np.dot(self.A, self.Sigma), self.A.T) + self.Q1  # predicted state covariance
        

    def wraptopi(self, x):
        # wraps angles in x, in radians, to the interval [−pi, pi] such that pi maps to pi and −pi maps to −pi. 
        # In general, odd, positive multiples of pi map to pi and odd, negative multiples of pi map to −pi.
        pi = np.pi
        x = x - np.floor(x/(2*pi)) *2 *pi
        # print('x2', type(x))
        if x>=pi:
            x = x-2*pi
        else:
            x = x
        
        return float(x)


    def correction_batch(self, z_stack, landmark1, landmark2, landmark3):
        """
        :param z_stack: inputted stacked measurments
        :return:
        """
        # z_hat1 = self.h1(self.Kf_1, self.x1, self.C_1)
        z_hat1 = self.h(self.x_pred1, landmark1[0], landmark1[1])
        z_hat2 = self.h(self.x_pred1, landmark2[0], landmark2[1])
        z_hat3 = self.h(self.x_pred1, landmark3[0], landmark3[1])

        H1=self.H(landmark1[0], landmark1[1], self.x_pred1, z_hat1)
        H2=self.H(landmark2[0], landmark2[1], self.x_pred1, z_hat2)
        H3=self.H(landmark3[0], landmark3[1], self.x_pred1, z_hat3)

        # stack
        # H=[H_1; H_2]; % 4*3
        H = np.vstack((H1, H2, H3))
        # Qstack=blkdiag(obj.Q, obj.Q); % 4*4 R_stack

        # S=H*obj.Sigma_pred*H' + Qstack; % 4*4
        self.S = np.dot(np.dot(H, self.Sigma_pred1), H.T) + self.R_stack
        # K= obj.Sigma_pred*H'*inv(S); % 3*4
        self.K = np.dot(np.dot(self.Sigma_pred1, H.T), np.linalg.inv(self.S))

        # self.z_hat_stack = np.append(z_hat1, z_hat2, axis=0)

        # self.v = z_stack - self.z_hat_stack # innovation
        diff1 = float(z_stack[0, 0] - z_hat1[0, 0])
        diff2 = float(z_stack[2, 0] - z_hat2[0, 0])
        diff3 = float(z_stack[4, 0] - z_hat3[0, 0])

        self.v = np.array([[self.wraptopi(diff1)],
                           [z_stack[1,0] - z_hat1[1,0]],
                           [self.wraptopi(diff2)],
                           [z_stack[3,0] - z_hat2[1,0]],
                           [self.wraptopi(diff3)],
                           [z_stack[5,0] - z_hat3[1,0]]])

        self.x1 = self.x_pred1 + np.dot(self.K, self.v)                     #note that x
        print('x[2]', self.x1)
        self.x1[2,0] = self.wraptopi(self.x1[2,0])

        I = np.eye(np.shape(self.x1)[0])
        temp = I - np.dot(self.K, H)
        self.Sigma = np.dot(np.dot(temp, self.Sigma_pred1), temp.T) + np.dot(np.dot(self.K, self.R_stack), self.K.T)
