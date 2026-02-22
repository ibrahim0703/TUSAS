import numpy as np


class ESKF:
    def __init__(self):
        # Durum Vektörleri (State Vectors)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = np.eye(3)

        # Bias (Hata) Vektörleri
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.g = np.array([0, 0, -9.81])

        # Q Matrisi: Sistemin/IMU'nun Kendi İç Gürültüsü (Process Noise)
        # Değerler EuRoC ADIS16448 IMU sensor.yaml dosyasından alınmıştır.
        self.Q = np.zeros((15, 15))
        self.Q[0:3, 0:3] = np.eye(3) * (0.001)  # Konum gürültüsü
        self.Q[3:6, 3:6] = np.eye(3) * (2.0e-3) ** 2  # İvmeölçer White Noise (a_noise_density)
        self.Q[6:9, 6:9] = np.eye(3) * (1.69e-4) ** 2  # Jiroskop White Noise (g_noise_density)
        self.Q[9:12, 9:12] = np.eye(3) * (1.93e-5) ** 2  # Jiroskop Bias Kayması (g_random_walk)
        self.Q[12:15, 12:15] = np.eye(3) * (3.0e-3) ** 2  # İvmeölçer Bias Kayması (a_random_walk)

        # R Matrisi: Kameranın (Ölçümün) Güvensizlik Matrisi
        # Telemetri CSV testimizdeki 0.15 m/s'lik ortalama sapmaya göre güncellendi.
        self.R_cam = np.diag([0.15 ** 2, 0.15 ** 2, 0.15 ** 2])

        # P Matrisi: Başlangıç Güvenilirliği (Kovaryans)
        self.P = np.eye(15) * 0.1
        self.P[3:6, 3:6] = np.eye(3) * 1.0  # Başlangıç hızı meçhul, güvensizlik yüksek

    def initialize_system(self, accel_samples, gyro_samples):
        # Statik Bias ve Yerçekimi Hizalaması
        self.bg = np.mean(gyro_samples, axis=0)
        a_mean = np.mean(accel_samples, axis=0)
        z_axis = a_mean / np.linalg.norm(a_mean)

        x_axis = np.array([1.0, 0.0, 0.0])
        x_axis = x_axis - z_axis * np.dot(x_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        self.R = np.vstack((x_axis, y_axis, z_axis)).T
        print("[ESKF] Yercekimi Hizalamasi ve Bias Baslatmasi Tamamlandi!")

    def predict(self, accel, gyro, dt):
        if dt <= 0: return

        # Gerçek ivme ve açısal hızı bias'lardan arındır
        accel_true = accel - self.ba
        gyro_true = gyro - self.bg

        # İvmeyi Dünya (World) eksenine çevir ve yerçekimini ekle
        accel_world = self.R @ accel_true + self.g

        # Fiziksel Kinematik Denklemleri
        self.p = self.p + self.v * dt + 0.5 * accel_world * (dt ** 2)
        self.v = self.v + accel_world * dt

        # Rotasyon Güncellemesi (Rodrigues Formülü)
        theta = np.linalg.norm(gyro_true) * dt
        if theta > 1e-8:
            axis = gyro_true / np.linalg.norm(gyro_true)
            K_skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R_delta = np.eye(3) + np.sin(theta) * K_skew + (1 - np.cos(theta)) * (K_skew @ K_skew)
            self.R = self.R @ R_delta

        # Hata Durumu (Error-State) Jacobian Matrisi (F)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        # P Matrisini Geleceğe Taşı (Kovaryans Yayılımı)
        self.P = F @ self.P @ F.T + self.Q

    def update_velocity(self, measured_velocity):
        # İnovasyon: Kameranın ölçtüğü hız ile IMU'nun tahmin ettiği hız arasındaki fark
        innovation = measured_velocity.flatten() - self.v

        # H (Gözlem) Matrisi: Biz sadece hızı (3:6 indeksleri) ölçüyoruz
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)

        # Kalman Kazancı (K) Hesaplaması
        S = H @ self.P @ H.T + self.R_cam
        K_gain = self.P @ H.T @ np.linalg.inv(S)

        # Hata Durumunu (Error State) Hesapla
        error_state = K_gain @ innovation

        # Gerçek Durumları Düzelt (Nominal State Update)
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        self.bg += error_state[9:12]  # Dinamik Gyro Bias Düzeltmesi
        self.ba += error_state[12:15]  # Dinamik Accel Bias Düzeltmesi

        # P Matrisini Güncelle (Güveni Artır)
        self.P = (np.eye(15) - K_gain @ H) @ self.P

    def get_speed(self):
        return np.linalg.norm(self.v)
