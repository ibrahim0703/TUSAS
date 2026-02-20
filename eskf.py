import numpy as np

class ESKF:
    def __init__(self):
        self.p = np.zeros(3) 
        self.v = np.zeros(3) 
        self.R = np.eye(3)   
        
        self.bg = np.zeros(3) 
        self.ba = np.zeros(3) 
        self.g = np.array([0, 0, -9.81]) 

        # Kovaryans ve Gürültü Matrisleri
        self.P = np.eye(15) * 0.1 
        self.Q = np.eye(15) * 0.001
        
        # Kameraya Güven Katsayısı (Sağırlaştırdık ki hatalara fırlamasın)
        self.R_cam = np.diag([5.0, 5.0, 10.0]) 

    def predict(self, accel, gyro, dt):
        if dt <= 0: return
        accel_true = accel - self.ba
        gyro_true = gyro - self.bg
        
        # Fizik Entegrasyonu
        accel_world = self.R @ accel_true + self.g
        self.p = self.p + self.v * dt + 0.5 * accel_world * (dt ** 2)
        self.v = self.v + accel_world * dt
        
        # Yönelim Güncellemesi
        theta = np.linalg.norm(gyro_true) * dt
        if theta > 1e-8:
            axis = gyro_true / np.linalg.norm(gyro_true)
            K_skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R_delta = np.eye(3) + np.sin(theta) * K_skew + (1 - np.cos(theta)) * (K_skew @ K_skew)
            self.R = self.R @ R_delta

        # State Transition (F Matrisi) - Pozisyon, hızdan etkilenir
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Belirsizliği İlerlet
        self.P = F @ self.P @ F.T + self.Q

    def update_velocity(self, measured_velocity):
        """
        Kameradan DÜNYA EKSENİNDE (World Frame) gelen 3 Boyutlu Hız ile filtreyi düzeltir.
        """
        innovation = measured_velocity.flatten() - self.v
        
        # H (Gözlem) Matrisi Hıza (3:6) bakar
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        
        # Kalman Kazancı
        S = H @ self.P @ H.T + self.R_cam 
        K_gain = self.P @ H.T @ np.linalg.inv(S) 
        
        error_state = K_gain @ innovation
        
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        
        self.P = (np.eye(15) - K_gain @ H) @ self.P

    def get_speed(self):
        return np.linalg.norm(self.v)
