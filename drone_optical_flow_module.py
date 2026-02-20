import numpy as np

class ESKF:
    def __init__(self):
        print("[SİSTEM] Error-State Kalman Filter (ESKF) Başlatıldı.")
        
        # ---------------------------------------------------------
        # 1. NOMİNAL DURUM (Nominal State) - Gerçekte Neredeyiz?
        # ---------------------------------------------------------
        self.p = np.zeros(3) # Pozisyon (X, Y, Z) - Metre
        self.v = np.zeros(3) # Hız (Vx, Vy, Vz) - m/s
        self.R = np.eye(3)   # Yönelim (Rotation Matrix) - 3x3 Birim Matris
        
        # IMU Fabrika Hataları (Bias)
        self.bg = np.zeros(3) # Jiroskop Bias (rad/s)
        self.ba = np.zeros(3) # İvmeölçer Bias (m/s^2)
        
        self.g = np.array([0, 0, -9.81]) # Yerçekimi Vektörü (Z ekseninde aşağı doğru)

        # ---------------------------------------------------------
        # 2. HATA DURUMU KOVARYANSI (Error-State Covariance - P)
        # ---------------------------------------------------------
        # Sistemimize ne kadar güveniyoruz? (15x15 Matris)
        # Sırası: [delta_p, delta_v, delta_theta, delta_bg, delta_ba]
        self.P = np.eye(15) * 0.01 
        
        # ---------------------------------------------------------
        # 3. GÜRÜLTÜ MATRİSLERİ (Noise)
        # ---------------------------------------------------------
        # IMU ne kadar gürültülü? (Q Matrisi)
        self.Q = np.eye(12) * 0.001 
        
        # Kamera (PnP) ne kadar gürültülü? (R Matrisi - Ölçüm gürültüsü)
        # Z ekseni (Derinlik) her zaman daha gürültülüdür, ona daha az güveniriz.
        self.R_cam = np.diag([0.05, 0.05, 0.2]) 

    def predict(self, accel, gyro, dt):
        """
        Adım 1: TAHMİN (Prediction) - Yüksek Frekans (IMU verisi geldikçe çalışır)
        Fiziksel formüllerle (İvme -> Hız -> Konum) bir sonraki adımı tahmin ederiz.
        """
        # 1. Bias düzeltmesi (Fabrika hatalarını çıkar)
        accel_true = accel - self.ba
        gyro_true = gyro - self.bg
        
        # 2. Nominal Durumu Güncelle (Fizik Entegrasyonu)
        # Yeni Pozisyon: p = p + v*dt + 0.5*(R*a + g)*dt^2
        accel_world = self.R @ accel_true + self.g
        self.p = self.p + self.v * dt + 0.5 * accel_world * (dt ** 2)
        
        # Yeni Hız: v = v + (R*a + g)*dt
        self.v = self.v + accel_world * dt
        
        # Yeni Yönelim: Basit Rodrigues Dönüşümü (Daha profesyoneli Quaternion'dur)
        # R = R * exp(w*dt)
        theta = np.linalg.norm(gyro_true) * dt
        if theta > 1e-8:
            axis = gyro_true / np.linalg.norm(gyro_true)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R_delta = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            self.R = self.R @ R_delta

        # 3. Kovaryansı (Belirsizliği) Güncelle: P = F * P * F^T + Q
        # (Şimdilik F matrisinin karmaşık Jacobian hesaplamalarını atlıyoruz, 
        # belirsizliğin zamanla arttığını simüle ediyoruz).
        self.P = self.P + np.eye(15) * 0.005 # Her adımda belirsizlik artar

    def update_with_pnp(self, pnp_translation):
        """
        Adım 2: GÜNCELLEME (Correction) - Düşük Frekans (Kamera Keyframe ürettikçe çalışır)
        Kameradan gelen gerçek ölçümle, IMU'nun tahminini çarpıştırır.
        """
        # Hata (Innovation): Kameranın gördüğü pozisyon eksi bizim IMU ile tahmin ettiğimiz pozisyon
        innovation = pnp_translation.flatten() - self.p
        
        # Kalman Kazancı (Kalman Gain - K) hesaplaması
        # H (Gözlem Matrisi) pozisyonu izlediğimiz için ilk 3x3'lük kısımdır.
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        
        S = H @ self.P @ H.T + self.R_cam # İnovasyon Kovaryansı
        K = self.P @ H.T @ np.linalg.inv(S) # K = P * H^T * S^-1
        
        # Hata Durumunu (Error State) Hesapla
        error_state = K @ innovation
        
        # Nominal Duruma Hataları Enjekte Et (Düzeltme)
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        # (Yönelim ve Bias düzeltmeleri bu aşamada sade tutulmuştur)
        
        # Belirsizliği (Kovaryans) Düşür: P = (I - K*H) * P
        self.P = (np.eye(15) - K @ H) @ self.P
