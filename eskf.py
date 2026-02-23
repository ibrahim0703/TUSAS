import numpy as np

def skew(x):
    """3D vektörü Skew-Symmetric (Çapraz Çarpım) matrisine çevirir."""
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

class ESKF:
    def __init__(self):
        # --- NOMİNAL DURUM (NOMINAL STATE) ---
        self.p = np.zeros(3) 
        self.v = np.zeros(3) 
        self.R = np.eye(3)   
        
        self.bg = np.zeros(3) 
        self.ba = np.zeros(3) 
        self.g = np.array([0, 0, -9.81]) 
        
        # --- KOVARYANS VE GÜRÜLTÜ MATRİSLERİ ---
        self.P = np.eye(15) * 0.1 
        self.P[3:6, 3:6] = np.eye(3) * 1.0 
        
        # Süreç Gürültüsü (EuRoC Makalesi Değerleri)
        self.Q = np.zeros((15, 15))
        self.Q[0:3, 0:3] = np.eye(3) * (0.001)        
        self.Q[3:6, 3:6] = np.eye(3) * (2.0e-3)**2    
        self.Q[6:9, 6:9] = np.eye(3) * (1.69e-4)**2   
        self.Q[9:12, 9:12] = np.eye(3) * (1.93e-5)**2 
        self.Q[12:15, 12:15] = np.eye(3) * (3.0e-3)**2

        # FİZİKSEL DÜZELTME: Artık kameradan KONUM (Metre) alıyoruz. PnP'nin sapma payı 5 santimdir.
        self.R_cam = np.diag([0.05**2, 0.05**2, 0.05**2]) 

    def initialize_system(self, accel_samples, gyro_samples):
        self.bg = np.mean(gyro_samples, axis=0)
        a_mean = np.mean(accel_samples, axis=0)
        
        z_axis = a_mean / np.linalg.norm(a_mean) 
        x_axis = np.array([1.0, 0.0, 0.0])
        x_axis = x_axis - z_axis * np.dot(x_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        self.R = np.vstack((x_axis, y_axis, z_axis)).T
        print("[ESKF] Tam Donanimli Hizalama ve Bias Baslatmasi Tamamlandi!")

    def predict(self, accel, gyro, dt):
        if dt <= 0: return
        
        accel_true = accel - self.ba
        gyro_true = gyro - self.bg
        accel_world = self.R @ accel_true + self.g
        
        # 1. Nominal State Kinematiği
        self.p = self.p + self.v * dt + 0.5 * accel_world * (dt ** 2)
        self.v = self.v + accel_world * dt
        
        theta_vec = gyro_true * dt
        theta_mag = np.linalg.norm(theta_vec)
        if theta_mag > 1e-8:
            axis = theta_vec / theta_mag
            K_skew = skew(axis)
            R_delta = np.eye(3) + np.sin(theta_mag) * K_skew + (1 - np.cos(theta_mag)) * (K_skew @ K_skew)
            self.R = self.R @ R_delta

        # 2. Hata Jacobian (F) Matrisi - JOAN SOLA DENKLEM 269 UYARLAMASI
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt                           
        F[3:6, 6:9] = -self.R @ skew(accel_true) * dt          
        F[3:6, 12:15] = -self.R * dt                           
        F[6:9, 9:12] = -np.eye(3) * dt                         
        
        # Sola Eq 269: dtheta_k+1 = R^T{w*dt} dtheta_k
        if theta_mag > 1e-8:
            # R_delta'nın transpozu
            R_delta_T = np.eye(3) - np.sin(theta_mag) * K_skew + (1 - np.cos(theta_mag)) * (K_skew @ K_skew)
            F[6:9, 6:9] = R_delta_T

        self.P = F @ self.P @ F.T + self.Q

    def update_position(self, measured_position):
        # FİZİKSEL DÜZELTME: İnovasyon artık hız farkı değil, doğrudan Dünya üzerindeki KONUM FARKIDIR.
        innovation = measured_position.flatten() - self.p
        
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3) # Sadece konumu (0, 1, 2) ölçüyoruz.
        
        S = H @ self.P @ H.T + self.R_cam 
        K_gain = self.P @ H.T @ np.linalg.inv(S) 
        
        error_state = K_gain @ innovation
        
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        
        dtheta = error_state[6:9]
        dtheta_mag = np.linalg.norm(dtheta)
        if dtheta_mag > 1e-8:
            axis = dtheta / dtheta_mag
            K_skew = skew(axis)
            R_delta = np.eye(3) + np.sin(dtheta_mag) * K_skew + (1 - np.cos(dtheta_mag)) * (K_skew @ K_skew)
            self.R = self.R @ R_delta
            
        self.bg += error_state[9:12]
        self.ba += error_state[12:15]
        
        self.P = (np.eye(15) - K_gain @ H) @ self.P
        self.P = (self.P + self.P.T) / 2.0 

    def get_speed(self):
        return np.linalg.norm(self.v)
