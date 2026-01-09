import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. BASİT KALMAN FİLTRESİ SINIFI (Linear KF)
# ==========================================
class SimpleKF:
    def __init__(self, dt, init_x, process_std):
        self.dt = dt
        # State: [x, y, z, vx, vy, vz] (6x1)
        self.x = init_x.copy()
        
        # Fizik Matrisi (F)
        # p = p + v*dt
        self.F = np.eye(6)
        self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt
        
        # Kovaryans (P) - Başlangıç belirsizliği
        self.P = np.eye(6) * 10.0
        
        # Process Noise (Q) - Sistemin titremesi (İvme gürültüsü)
        # Bu matris, modelimizin %100 doğru olmadığını kabul eder.
        q_noise = process_std ** 2
        self.Q = np.eye(6) * q_noise

    def predict(self):
        # x = F * x
        self.x = np.dot(self.F, self.x)
        # P = F * P * F_transpose + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z, H, R):
        """
        Esnek Update Fonksiyonu.
        z: Ölçüm vektörü (N x 1)
        H: Ölçüm Matrisi (N x 6) - Hangi state ölçüldü?
        R: Ölçüm Gürültüsü (N x N)
        """
        # y = z - H * x (Innovation: Ölçüm ile Tahmin arasındaki fark)
        y = z - np.dot(H, self.x)
        
        # S = H * P * H_transpose + R (Innovation Covariance)
        S = np.dot(np.dot(H, self.P), H.T) + R
        
        # K = P * H_transpose * inv(S) (Kalman Gain)
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        
        # x = x + K * y (State Update)
        self.x = self.x + np.dot(K, y)
        
        # P = (I - K * H) * P (Covariance Update)
        I = np.eye(6)
        self.P = np.dot((I - np.dot(K, H)), self.P)

# ==========================================
# 2. VERİ ÜRETİMİ VE AYARLAR
# ==========================================
dt = 0.05               # 20 Hz
total_steps = 400       # 20 saniye
gps_interval = 20       # Case 3 için: GPS her 20 adımda bir gelsin (1 Hz)

# Gürültü Standart Sapmaları
accel_noise_std = 0.5   # Process Noise (Sarsıntı)
odo_noise_std = 0.2     # Hız ölçüm hatası (m/s)
gps_noise_std = 3.0     # GPS konum hatası (metre)

# Matrislerin Hazırlanması (Update fonksiyonuna göndermek için)
# H_odo: Sadece Hızları (vx, vy, vz) ölçer (Son 3 eleman)
H_odo = np.zeros((3, 6))
H_odo[0, 3] = 1; H_odo[1, 4] = 1; H_odo[2, 5] = 1
R_odo = np.eye(3) * (odo_noise_std**2)

# H_gps: Sadece Konumları (x, y, z) ölçer (İlk 3 eleman)
H_gps = np.zeros((3, 6))
H_gps[0, 0] = 1; H_gps[1, 1] = 1; H_gps[2, 2] = 1
R_gps = np.eye(3) * (gps_noise_std**2)

# --- Ground Truth Oluşturma ---
true_pos = np.zeros((total_steps, 3))
true_vel = np.zeros((total_steps, 3))

# Başlangıç
p = np.array([0., 0., 0.]).reshape(6,1) # State vektörü
v = np.array([5., 2., 0.5]) # Sabit hız isteği

for i in range(total_steps):
    # Rastgele İvme (Process Noise)
    noise_a = np.random.normal(0, accel_noise_std, 3)
    
    # Kinematik: p = p + v*dt + 0.5*a*dt^2
    p[0:3, 0] += v * dt + 0.5 * noise_a * dt**2
    p[3:6, 0] = v + noise_a * dt # Hızı gürültüyle güncelle
    v = p[3:6, 0] # Yeni hızı kaydet
    
    true_pos[i] = p[0:3, 0]
    true_vel[i] = p[3:6, 0]

# --- Sensör Ölçümleri (Simülasyon) ---
meas_odo = true_vel + np.random.normal(0, odo_noise_std, true_vel.shape)
meas_gps = true_pos + np.random.normal(0, gps_noise_std, true_pos.shape)

# ==========================================
# 3. FİLTRELERİ ÇALIŞTIR (3 CASE)
# ==========================================
# 3 tane ayrı filtre başlatıyoruz
kf1 = SimpleKF(dt, np.zeros((6,1)), accel_noise_std) # Case 1: Sadece Odo
kf2 = SimpleKF(dt, np.zeros((6,1)), accel_noise_std) # Case 2: Odo + GPS (20Hz)
kf3 = SimpleKF(dt, np.zeros((6,1)), accel_noise_std) # Case 3: Odo + GPS (1Hz)

# Sonuçları saklamak için
res_kf1 = []; res_kf2 = []; res_kf3 = []

print("Simülasyon Başladı...")

for i in range(total_steps):
    # --- ORTAK ADIM: PREDICT ---
    # Hepsi önce fiziği kullanarak tahmin yürütür
    kf1.predict()
    kf2.predict()
    kf3.predict()
    
    # --- SENSÖR UPDATE ---
    
    # ODOMETRE (20 Hz) - Her adımda var
    z_odo = meas_odo[i].reshape(3, 1)
    kf1.update(z_odo, H_odo, R_odo) # Case 1: Sadece bunu kullanır
    kf2.update(z_odo, H_odo, R_odo) # Case 2: Hem bunu...
    kf3.update(z_odo, H_odo, R_odo) # Case 3: Hem bunu kullanır
    
    # GPS (Duruma göre)
    z_gps = meas_gps[i].reshape(3, 1)
    
    # CASE 2: GPS her zaman var (20 Hz)
    kf2.update(z_gps, H_gps, R_gps) 
    
    # CASE 3: GPS sadece belli aralıklarla var (1 Hz)
    if i % gps_interval == 0:
        kf3.update(z_gps, H_gps, R_gps) # Düzeltme vuruşu!
        
    # --- KAYIT ---
    res_kf1.append(kf1.x[0:3, 0].copy())
    res_kf2.append(kf2.x[0:3, 0].copy())
    res_kf3.append(kf3.x[0:3, 0].copy())

# Array'e çevir
res_kf1 = np.array(res_kf1)
res_kf2 = np.array(res_kf2)
res_kf3 = np.array(res_kf3)

# ==========================================
# 4. GÖRSELLEŞTİRME (ACIMASIZ GERÇEKLER)
# ==========================================
plt.figure(figsize=(14, 6))

# Sadece X-Y Düzlemine bakalım (Kuş bakışı)
plt.plot(true_pos[:,0], true_pos[:,1], 'k-', linewidth=2, label='Gerçek Rota (Ground Truth)')

# Case 1: Sadece Odo
plt.plot(res_kf1[:,0], res_kf1[:,1], 'r--', label='Case 1: Sadece Odo (Drift)')

# Case 2: Odo + Full GPS
plt.plot(res_kf2[:,0], res_kf2[:,1], 'g-', alpha=0.6, linewidth=1, label='Case 2: Odo + 20Hz GPS (Mükemmel)')

# Case 3: Odo + Seyrek GPS
plt.plot(res_kf3[:,0], res_kf3[:,1], 'b.-', markersize=3, markevery=gps_interval, label='Case 3: Odo + 1Hz GPS (Düzeltmeli)')

plt.title('3 Senaryonun Kıyaslaması: Drift vs Füzyon')
plt.xlabel('X Konumu (m)')
plt.ylabel('Y Konumu (m)')
plt.legend()
plt.grid()

# Hata Analizi (Zaman Grafiği - X ekseni Hatası)
plt.figure(figsize=(14, 4))
err1 = res_kf1[:,0] - true_pos[:,0]
err3 = res_kf3[:,0] - true_pos[:,0]
plt.plot(err1, 'r', label='Case 1 Hatası (Sürekli Artıyor)')
plt.plot(err3, 'b', label='Case 3 Hatası (GPS geldikçe sıfırlanıyor)')
plt.title('Hata Analizi (X Ekseni)')
plt.xlabel('Zaman Adımı')
plt.ylabel('Hata (metre)')
plt.legend()
plt.grid()

plt.show()
