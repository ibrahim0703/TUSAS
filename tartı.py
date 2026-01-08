import numpy as np
import matplotlib.pyplot as plt

# --- 1. SİMÜLASYON PARAMETRELERİ ---
DURATION = 100 # iteration
DT = 1  # step size
TRUE_MASS = 130  # gerçek ağırlık
START_MASS = 140  # başlangıç ağırlık tahmini

# Sensör Hatası
MASS_PRED_NOISE_STD = 5  # GPS ne kadar titriyor? (Standart sapma - Metre)


# --- 2. SINIFLAR (KALMAN FİLTRESİ MANTIĞI) ---
class SimpleKalmanFilter:
    def __init__(self, initial_m, process_noise, measure_noise,p0):
        # State (Durum)
        self.m = initial_m

        # Covariance Matrix (P): Kendimize ne kadar güveniyoruz?
        # modelin belirsizliğini belirler her stepte güncellenir.

        self.P = p0

        # State Transition (F): Fizik kuralı m = m
        # system model / her adımda bizim tartının ölçümünün matematiksel modeli m = m 
        self.F = 1

        # Measurement Matrix (H): state vektörünün boyutunu günceller her zaman m oldugu için boyut değişmez.
        self.H = 1

        self.Q = process_noise  # Process noise covariance (uncertainty in the process). m değeri sürekli aynı olmalı modelimize göre m değişmiyor o yüzden Q = 0 olmalı.
        self.R = measure_noise  # Measurement noise covariance (uncertainty in the measurements). tartıdaki hatanın standart sapmasının karesi (covariance)

    def predict(self):
        """
        tartı tahmini adımı: Sadece fiziğe güvenerek yeni ağırlığı tahmin et.
        Matematik: m = m / bu yüzden değişmeyecek.
        """
        self.m = self.F * self.m
        self.P = self.P * self.F + self.Q
        # P matrixini güncellemek için burada yazılan dotlar hep bu tarzda yazılıyor.
        return self.m  # Tahmini tartı o anki değer

    def update(self, z):
        """
        ölçüm adımı: Dışarıdan gelen ölçümle (z) tahmini düzelt.

        """
        # 1. Hata ne kadar? (Innovation)
        y = z - self.H * self.m

        # tahmin - o anki değer

        # 2. Kalman Kazancı (K): Kime güveneceğiz? tartıya mı , mevcut ölçüme mi?
        K = self.P / (self.P+self.R)

        # 3. Durumu Güncelle (Correction)
        self.m = self.m + K*y

        # 4. Güvenimizi Güncelle (Covariance Update)
        self.P = (1-K) * self.P

        return self.m  # Düzeltilmiş konum


# --- 3. SİMÜLASYONU ÇALIŞTIR ---
time_steps = np.arange(0, DURATION, DT)
true_pos = []
tartı_measurements = []
current_mass = []  # Kalman olmadan sadece INS (Dead Reckoning)
kalman_estimates = []

# Gerçek Dünya Değişkenleri
current_true_pos = TRUE_MASS
current_ins_pos = START_MASS

# Kalman Başlat

kf = SimpleKalmanFilter(initial_m = START_MASS, process_noise = 0, measure_noise = MASS_PRED_NOISE_STD ** 2, p0 = 0.1)

print("Simülasyon Başlıyor...")

for t in time_steps:
    # A. GERÇEK DÜNYA (TRUTH)
    current_ins_pos = current_ins_pos # bi değişiklik yok tartı durumu hep aynı
    # B. SENSÖR SİMÜLASYONU (GÜRÜLTÜ EKLEME)
    # 1. TARTI AĞIRLIĞINA: Gerçek konuma rastgele hata ekle
    z_tartı = current_true_pos + np.random.normal(0, MASS_PRED_NOISE_STD)


    # C. KALMAN FİLTRESİ
    kf.predict()  # Önce tahmin et bişey değişmez tartı değeri hep aynı model m = m.
    est_pos = kf.update(z_tartı)  # Sonra tartı ölçümüyle düzelt.

    # Verileri Kaydet
    true_pos.append(current_true_pos) # gerçek ağırlık değeri değişmez 
    tartı_measurements.append(z_tartı)    # tartı mesarument sonuçları her stepte değişir.
    current_mass.append(current_ins_pos) # her adımdaki güncellenmiş state m değeri 
    kalman_estimates.append(est_pos)     # kalman estimation

# --- 4. GÖRSELLEŞTİRME ---
plt.figure(figsize=(12, 6))

# Hatayı görmek için 'Gerçek Konum'dan farklarını çizelim (Daha çarpıcı olur)
plt.plot(time_steps, np.array(tartı_measurements) - np.array(true_pos),
         'g.', alpha=0.3, label='GPS Ölçümü (Gürültülü)')

plt.plot(time_steps, np.array(current_mass) - np.array(true_pos),
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
