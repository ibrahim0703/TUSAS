import numpy as np
import matplotlib.pyplot as plt

# --- 1. SİMÜLASYON PARAMETRELERİ ---
DURATION = 50  # Saniye
DT = 0.1  # Zaman adımı
TRUE_VELOCITY = 100  # Uçağın gerçek hızı (m/s)
START_POS = 0  # Başlangıç konumu

# Sensör Hataları (Acımasız Gerçekler)
GPS_NOISE_STD = 25  # GPS ne kadar titriyor? (Standart sapma - Metre)
INS_BIAS_DRIFT = 0.5  # INS her saniye ne kadar hata biriktiriyor? (m/s hata)
ACCEL_NOISE = 0.2  # İvmeölçerdeki anlık gürültü


# --- 2. SINIFLAR (KALMAN FİLTRESİ MANTIĞI) ---
class SimpleKalmanFilter:
    def __init__(self, initial_x, initial_v, process_noise, measure_noise):
        # State (Durum): [Konum, Hız]
        self.x = np.array([initial_x, initial_v])

        # Covariance Matrix (P): Kendimize ne kadar güveniyoruz?
        # Başta belirsiziz, o yüzden büyük sayılar.
        self.P = np.eye(2) * 1000

        # State Transition (F): Fizik kuralı (x = x + v*dt)
        self.F = np.array([[1, DT],
                           [0, 1]])

        # Measurement Matrix (H): Neyi ölçüyoruz? Sadece Konumu (GPS).
        self.H = np.array([[1, 0]])

        self.Q = process_noise  # Sistem gürültüsü (Rüzgar vb.)
        self.R = measure_noise  # Ölçüm gürültüsü (GPS hatası)

    def predict(self):
        """
        INS ADIMI: Sadece fiziğe güvenerek yeni konumu tahmin et.
        Matematik: x_yeni = F * x_eski
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0]  # Tahmini konum

    def update(self, z):
        """
        GPS ADIMI: Dışarıdan gelen ölçümle (z) tahmini düzelt.
        Bu kısım sihrin olduğu yerdir.
        """
        # 1. Hata ne kadar? (Innovation)
        y = z - np.dot(self.H, self.x)

        # 2. Kalman Kazancı (K): Kime güveneceğiz? GPS'e mi, INS'e mi?
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 3. Durumu Güncelle (Correction)
        self.x = self.x + np.dot(K, y)

        # 4. Güvenimizi Güncelle (Covariance Update)
        I = np.eye(2)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.x[0]  # Düzeltilmiş konum


# --- 3. SİMÜLASYONU ÇALIŞTIR ---
time_steps = np.arange(0, DURATION, DT)
true_pos = []
gps_measurements = []
ins_estimates = []  # Kalman olmadan sadece INS (Dead Reckoning)
kalman_estimates = []

# Gerçek Dünya Değişkenleri
current_true_pos = START_POS
current_ins_pos = START_POS
ins_velocity_error = 0  # Başlangıçta hata yok ama artacak

# Kalman Başlat
kf = SimpleKalmanFilter(initial_x=0, initial_v=10,
                        process_noise=np.eye(2) * 1,
                        measure_noise=GPS_NOISE_STD ** 2)

print("Simülasyon Başlıyor...")

for t in time_steps:
    # A. GERÇEK DÜNYA (TRUTH)
    current_true_pos += TRUE_VELOCITY * DT

    # B. SENSÖR SİMÜLASYONU (GÜRÜLTÜ EKLEME)
    # 1. GPS: Gerçek konuma rastgele hata ekle
    z_gps = current_true_pos + np.random.normal(0, GPS_NOISE_STD)

    # 2. INS: Hız verisine 'bias' (kayma) ekle. Bu hata kümülatiftir!
    ins_velocity_error += INS_BIAS_DRIFT * DT * 0.1  # Hata yavaş yavaş artıyor
    perceived_velocity = TRUE_VELOCITY + ins_velocity_error + np.random.normal(0, ACCEL_NOISE)
    current_ins_pos += perceived_velocity * DT

    # C. KALMAN FİLTRESİ
    kf.predict()  # Önce tahmin et (INS mantığı)
    est_pos = kf.update(z_gps)  # Sonra GPS ile düzelt

    # Verileri Kaydet
    true_pos.append(current_true_pos)
    gps_measurements.append(z_gps)
    ins_estimates.append(current_ins_pos)
    kalman_estimates.append(est_pos)

# --- 4. GÖRSELLEŞTİRME ---
plt.figure(figsize=(12, 6))

# Hatayı görmek için 'Gerçek Konum'dan farklarını çizelim (Daha çarpıcı olur)
plt.plot(time_steps, np.array(gps_measurements) - np.array(true_pos),
         'g.', alpha=0.3, label='GPS Ölçümü (Gürültülü)')

plt.plot(time_steps, np.array(ins_estimates) - np.array(true_pos),
         'r--', linewidth=2, label='Sadece INS (Zamanla Sapar!)')

plt.plot(time_steps, np.array(kalman_estimates) - np.array(true_pos),
         'b-', linewidth=3, label='Kalman Filtresi (EGI)')

plt.title('Hata Analizi: Kalman Filtresi vs Tek Başına Sensörler')
plt.xlabel('Zaman (saniye)')
plt.ylabel('Hata (metre)')
plt.axhline(0, color='k', linestyle='-', alpha=0.5)  # Gerçek (Sıfır Hata) çizgisi
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()
