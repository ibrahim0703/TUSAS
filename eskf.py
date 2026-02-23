import numpy as np

def skew(x):
    """3D vektörü Skew-Symmetric (Çapraz Çarpım) matrisine çevirir. Fiziğin kalbidir."""
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

class ESKF:
    def __init__(self):
        # --- NOMİNAL DURUM (NOMINAL STATE) ---
        self.p = np.zeros(3) # Konum (Dünya ekseni)
        self.v = np.zeros(3) # Hız (Dünya ekseni)
        self.R = np.eye(3)   # Rotasyon (Gövdeden Dünyaya)
        
        self.bg = np.zeros(3) # Jiroskop Biası
        self.ba = np.zeros(3) # İvmeölçer Biası
        self.g = np.array([0, 0, -9.81]) # Yerçekimi Vektörü
        
        # --- KOVARYANS VE GÜRÜLTÜ MATRİSLERİ ---
        self.P = np.eye(15) * 0.1 
        self.P[3:6, 3:6] = np.eye(3) * 1.0 # Hız başlangıçta daha şüpheli
        
        self.Q = np.zeros((15, 15))
        self.Q[0:3, 0:3] = np.eye(3) * (0.001)        
        self.Q[3:6, 3:6] = np.eye(3) * (2.0e-3)**2    # İvmeölçer Noise
        self.Q[6:9, 6:9] = np.eye(3) * (1.69e-4)**2   # Jiroskop Noise
        self.Q[9:12, 9:12] = np.eye(3) * (1.93e-5)**2 # Jiroskop Random Walk
        self.Q[12:15, 12:15] = np.eye(3) * (3.0e-3)**2# İvmeölçer Random Walk

        # Kameranın hız ölçümündeki hata payı (Varyans)
        self.R_cam = np.diag([0.15**2, 0.15**2, 0.15**2]) 

    def initialize_system(self, accel_samples, gyro_samples):
        """Drone yerdeyken ilk hizalamayı (Z eksenini yerçekimine oturtma) yapar."""
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
        """IMU verisiyle nominal durumu ilerletir ve Hata Kovaryansını (P) büyütür."""
        if dt <= 0: return
        
        accel_true = accel - self.ba
        gyro_true = gyro - self.bg
        
        # İvmeyi Dünya eksenine çevir ve yerçekimini ekle
        accel_world = self.R @ accel_true + self.g
        
        # 1. Nominal State Kinematiği
        self.p = self.p + self.v * dt + 0.5 * accel_world * (dt ** 2)
        self.v = self.v + accel_world * dt
        
        theta = np.linalg.norm(gyro_true) * dt
        if theta > 1e-8:
            axis = gyro_true / theta
            K_skew = skew(axis)
            R_delta = np.eye(3) + np.sin(theta) * K_skew + (1 - np.cos(theta)) * (K_skew @ K_skew)
            self.R = self.R @ R_delta

        # 2. Hata Jacobian (F) Matrisi - İŞTE ÖNCEKİ KODDA EKSİK OLAN KALP BURASI
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt                           # Hız -> Konum
        F[3:6, 6:9] = -self.R @ skew(accel_true) * dt          # Rotasyon Hatası -> Hız Hatası (Yerçekimi sızıntısı)
        F[3:6, 12:15] = -self.R * dt                           # İvme Biası -> Hız
        F[6:9, 9:12] = -np.eye(3) * dt                         # Gyro Biası -> Rotasyon
        
        # Hata Kovaryansının (P) Zamanla Büyümesi
        self.P = F @ self.P @ F.T + self.Q

    def update_velocity(self, measured_velocity):
        """Kamera hızı getirdiğinde aradaki farkı bulur ve 15 hatayı birden düzeltir."""
        innovation = measured_velocity.flatten() - self.v
        
        # Sadece 3., 4., 5. indeksleri (Hız) ölçebiliyoruz
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        
        # Kalman Kazancı (Ne kadar Kameraya, Ne kadar IMU'ya güveneceğiz?)
        S = H @ self.P @ H.T + self.R_cam 
        K_gain = self.P @ H.T @ np.linalg.inv(S) 
        
        # 15 Boyutlu Hata Vektörünü (Faturayı) Çıkar
        error_state = K_gain @ innovation
        
        # --- NOMİNAL DURUMU DÜZELT ---
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        
        # İŞTE ESKİ KODDAKİ ÖLÜMCÜL EKSİK: Rotasyon (Attitude) Düzeltmesi
        dtheta = error_state[6:9]
        theta_mag = np.linalg.norm(dtheta)
        if theta_mag > 1e-8:
            axis = dtheta / theta_mag
            K_skew = skew(axis)
            R_delta = np.eye(3) + np.sin(theta_mag) * K_skew + (1 - np.cos(theta_mag)) * (K_skew @ K_skew)
            self.R = self.R @ R_delta # Rotasyonu hizala ki yerçekimi sızmasın!
            
        self.bg += error_state[9:12]
        self.ba += error_state[12:15]
        
        # Güven Matrisini (P) Küçült (Artık daha eminiz)
        self.P = (np.eye(15) - K_gain @ H) @ self.P

    def get_speed(self):
        return np.linalg.norm(self.v)
