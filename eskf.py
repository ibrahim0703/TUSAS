import numpy as np

class ESKF:
    def __init__(self):
        self.p = np.zeros(3) # Pozisyon
        self.v = np.zeros(3) # Hız
        self.R = np.eye(3)   # Yönelim
        
        self.bg = np.zeros(3) # Gyro Bias (Şimdilik statik)
        self.ba = np.zeros(3) # Accel Bias (Şimdilik statik)
        self.g = np.array([0, 0, -9.81]) 

        self.P = np.eye(15) * 0.1 
        self.R_cam = np.diag([0.1, 0.1, 0.5]) 

    def predict(self, accel, gyro, dt):
        if dt <= 0: return
        accel_true = accel - self.ba
        gyro_true = gyro - self.bg
        
        accel_world = self.R @ accel_true + self.g
        self.p = self.p + self.v * dt + 0.5 * accel_world * (dt ** 2)
        self.v = self.v + accel_world * dt
        
        theta = np.linalg.norm(gyro_true) * dt
        if theta > 1e-8:
            axis = gyro_true / np.linalg.norm(gyro_true)
            K_skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R_delta = np.eye(3) + np.sin(theta) * K_skew + (1 - np.cos(theta)) * (K_skew @ K_skew)
            self.R = self.R @ R_delta

        self.P = self.P + np.eye(15) * 0.001

    def update_with_pnp(self, pnp_translation, dt):
        innovation = pnp_translation.flatten() - self.p
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        
        S = H @ self.P @ H.T + self.R_cam 
        K_gain = self.P @ H.T @ np.linalg.inv(S) 
        
        error_state = K_gain @ innovation
        
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        self.P = (np.eye(15) - K_gain @ H) @ self.P

    def get_speed(self):
        return np.linalg.norm(self.v)
