import numpy as np

class ESKF:
    def __init__(self):
        self.p = np.zeros(3) 
        self.v = np.zeros(3) 
        self.R = np.eye(3)   
        
        self.bg = np.zeros(3) 
        self.ba = np.zeros(3) 
        self.g = np.array([0, 0, -9.81]) 

        # Kovaryans (Belirsizlik) Matrisi
        self.P = np.eye(15) * 0.1 
        
        # SİSTEM GÜRÜLTÜSÜ (Q) - IMU'ya ne kadar GÜVENMİYORUZ?
        self.Q = np.eye(15) * 0.001
        
        # ÖLÇÜM GÜRÜLTÜSÜ (R_cam) - PnP Kameraya ne kadar GÜVENMİYORUZ?
        # Sayıları devasa şekilde artırdım (0.1'den 5.0'a). 
        # Çünkü kamera gürültülüyse, filtre hızı aniden fırlatmamalıdır!
        self.R_cam = np.diag([5.0, 5.0, 10.0]) 

    def predict(self, accel, gyro, dt):
        if dt <= 0: return
        accel_true = accel - self.ba
        gyro_true = gyro - self.bg
        
        # 1. Fizik Entegrasyonu
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

        # -------------------------------------------------------------
        # 2. İŞTE EKSİK OLAN HAYATİ MATEMATİK: STATE TRANSITION (F MATRİSİ)
        # -------------------------------------------------------------
        # F matrisi, sistemin birbiriyle olan fiziksel ilişkisidir.
        F = np.eye(15)
        
        # Pozisyon (0:3), Hızdan (3:6) 'dt' kadar etkilenir. ( p = p + v*dt )
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Belirsizliği (Kovaryans) F matrisi ile ilerlet: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update_with_pnp(self, pnp_translation, dt):
        innovation = pnp_translation.flatten() - self.p
        
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        
        # Kalman Kazancı Hesaplama
        S = H @ self.P @ H.T + self.R_cam 
        K_gain = self.P @ H.T @ np.linalg.inv(S) 
        
        # Hata Durumunu Hesapla ve Enjekte Et
        error_state = K_gain @ innovation
        
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        
        # Belirsizliği Güncelle
        self.P = (np.eye(15) - K_gain @ H) @ self.P

    def get_speed(self):
        return np.linalg.norm(self.v)
