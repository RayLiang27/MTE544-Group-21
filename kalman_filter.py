

import numpy as np
M_PI=3.1415926535


# TODO Part 3: Comment the code explaining each part
class kalman_filter:
    
    # TODO Part 3: Initialize the covariances and the states    
    def __init__(self, P,Q,R, x, dt):
        
        self.P = P
        self.Q = Q
        self.R = R
        self.x = x # [x, y, theta, w, v, vdot]
        self.dt = dt
        
    # TODO Part 3: Replace the matrices with Jacobians where needed        
    def predict(self):

        self.A = self.jacobian_A()
        self.C = self.jacobian_H()
        
        self.motion_model()
        
        self.P= np.dot( np.dot(self.A, self.P), self.A.T) + self.Q

    # TODO Part 3: Calculate the Kalman gain and correct your estimates
    def update(self, z):

        
        kalman_gain = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
        
        self.x = self.x + kalman_gain @ (z - self.measurement_model())
        
        self.P = (np.eye(kalman_gain.shape[0], self.C.shape[1]) - kalman_gain @ self.C) @ self.P
        
    
    # TODO Part 3: Implement here the measurement model
    def measurement_model(self):
        x, y, th, w, v, vdot = self.x
        return np.array([
            v, # v
            w, # w
            vdot, # ax
            v * w, # ay
        ])
        
    # TODO Part 3: Impelment the motion model (state-transition matrice)
    def motion_model(self):
        
        x, y, th, w, v, vdot = self.x

        # if th <= -M_PI:
        #     th += 2*M_PI
        
        
        # elif th >= M_PI:
        #     th -= 2*M_PI

        dt = self.dt
        
        self.x = np.array([
            x + v * np.cos(th) * dt,
            y + v * np.sin(th) * dt,
            th + w * dt,
            w,
            v + vdot*dt,
            vdot,
        ])
        


    
    def jacobian_A(self):
        x, y, th, w, v, vdot = self.x
        dt = self.dt
        
        return np.array([
            #x, y,               th, w,                v, vdot
            [1, 0, -v*np.sin(th)*dt, 0,  np.cos(th) * dt,  0],  # = d(x + v * np.cos(th) * dt) /d col
            [0, 1,  v*np.cos(th)*dt, 0,  np.sin(th) * dt,  0],  # = d(y + v * np.sin(th) * dt) /d col
            [0, 0,                1, dt,               0,  0],  # = d(th + w * dt) /d col
            [0, 0,                0, 1,                0,  0],  # = d(w) /d col
            [0, 0,                0, 0,                1,  dt], # = d(v + vdot*dt) /d col
            [0, 0,                0, 0,                0,  1 ]  # = d(vdot) /d col
        ])
    
    # TODO Part 3: Implement here the jacobian of the H matrix (measurements)    
    def jacobian_H(self):
        x, y, th, w, v, vdot=self.x
        return np.array([
            #x, y,th, w, v,vdot
            [0,0,0  , 0, 1, 0],     # = d(v) /d col
            [0,0,0  , 1, 0, 0],     # = d(w) /d col
            [0,0,0  , 0, 0, 1],     # = d(vdot) /d col
            [0,0,0  , v, w, 0], # = d(v * w) /d col
        ])
        
    # TODO Part 3: return the states here    
    def get_states(self):
        return self.x
