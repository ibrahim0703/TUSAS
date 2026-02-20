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
        self.Q = np.eye(15) * 0.005 # IMU gürültüsü
        self.R_cam = np.diag([2.0, 2.0, 5.0]) # Kameraya güven katsayısı

    def initialize_system(self, accel_samples, gyro_samples):
        """
        [EKSİK OLAN GERÇEK FİZİK] 
        Filtre başlamadan önce ivmeölçerin okuduğu yerçekimi vektörünü bulup, 
        dünyanın neresinin "Aşağı" olduğunu matrislere öğretir.
        """
        # Gyro'nun fabrika hatasını (Bias) bul ve sıfırla
        self.bg = np.mean(gyro_samples, axis=0)
        
        # Yerçekimi vektörünün yönünü bul
        a_mean = np.mean(accel_samples, axis=0)
        z_axis = a_mean / np.linalg.norm(a_mean) # Z ekseni yerçekimine hizalandı
        
        # X ve Y eksenlerini ortogonal (dik) olarak oluştur
        x_axis = np.array([1.0, 0.0, 0.0])
        x_axis = x_axis - z_axis * np.dot(x_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Başlangıç Dönüş Matrisini (R) Kusursuzlaştır
        self.R = np.vstack((x_axis, y_axis, z_axis)).T
        print("[SİSTEM] Yerçekimi Hizalaması ve Bias Temizliği Tamamlandı!")

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
        
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        self.P = (np.eye(15) - K_gain @ H) @ self.P

    def get_speed(self):
        return np.linalg.norm(self.v)
