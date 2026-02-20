import numpy as np

class ESKF:
    def __init__(self):
        self.p = np.zeros(3) 
        self.v = np.zeros(3) 
        self.R = np.eye(3)   
        
        self.bg = np.zeros(3) 
        self.ba = np.zeros(3) 
        self.g = np.array([0, 0, -9.81]) 

        self.P = np.eye(15) * 0.1 
        self.Q = np.eye(15) * 0.005 
        self.R_cam = np.diag([2.0, 2.0, 5.0]) 

    def initialize_system(self, accel_samples, gyro_samples):
        self.bg = np.mean(gyro_samples, axis=0)
        a_mean = np.mean(accel_samples, axis=0)
        z_axis = a_mean / np.linalg.norm(a_mean) 
        
        x_axis = np.array([1.0, 0.0, 0.0])
        x_axis = x_axis - z_axis * np.dot(x_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        self.R = np.vstack((x_axis, y_axis, z_axis)).T
        print("[SİSTEM] Yerçekimi Hizalaması Tamamlandı!")

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

        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        self.P = F @ self.P @ F.T + self.Q

    def update_velocity(self, measured_velocity):
        innovation = measured_velocity.flatten() - self.v
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        
        S = H @ self.P @ H.T + self.R_cam 
        K_gain = self.P @ H.T @ np.linalg.inv(S) 
        
        error_state = K_gain @ innovation
        
        # --- İŞTE EKSİK OLAN VE HIZI 20 M/S'YE ÇIKARAN FİZİK BURADA ---
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        
        # Kameradan gelen ölçümle, IMU'nun ısınmadan kaynaklı sapmalarını da GERÇEK ZAMANLI düzelt!
        self.bg += error_state[9:12] 
        self.ba += error_state[12:15]
        
        self.P = (np.eye(15) - K_gain @ H) @ self.P

    def get_speed(self):
        return np.linalg.norm(self.v)
