import numpy as np
import matplotlib.pyplot as plt

# --- 1. SİMÜLASYON PARAMETRELERİ ---
freq = 20.0             # 20 Hz
dt = 1.0 / freq         # 0.05 saniye
total_time = 10.0       # 10 saniyelik veri
steps = int(total_time * freq)

# --- 2. SİSTEM GÜRÜLTÜ AYARLARI (Process Noise) ---
# Uçan/giden cisim dümdüz gitmez, hava akımı veya motor titremesi olur.
# Bu "Gaussian" ivme gürültüsü, Kalman'ın "Q" matrisinin temelidir.
accel_noise_std = 0.5   # m/s^2 (Rastgele sarsıntı miktarı)

# --- 3. BAŞLANGIÇ DURUMU (Ground Truth) ---
# State: [x, y, z, vx, vy, vz]
true_pos = np.array([0.0, 0.0, 0.0])  # Başlangıç Konumu
true_vel = np.array([10.0, 5.0, 0.0]) # Başlangıç Hızı (Çapraz gidiyor)

# Verileri saklamak için depolar
log_true_pos = []
log_true_vel = []
log_odo_meas = [] # Odometre (Hız) Ölçümü

# --- 4. SİMÜLASYON DÖNGÜSÜ ---
print(f"Simülasyon Başlıyor: {freq}Hz, {total_time}sn")

for i in range(steps):
    # --- A. FİZİK MOTORU (Gerçek Hareket) ---
    
    # 1. Kontrol Girdisi (İvme):
    # Diyelim ki pilot hafifçe yukarı ve sağa gitmek istiyor (Sabit İvme)
    # AMA: Buna rastgele "Gaussian" gürültü biniyor.
    control_accel = np.array([0.5, -0.2, 0.1]) # Ana komut
    process_noise = np.random.normal(0, accel_noise_std, 3) # Gürültü
    
    true_accel = control_accel + process_noise # Gerçek ivme
    
    # 2. Kinematik (Integral Alıyoruz)
    # Önce konumu güncelle (Eski hızla)
    true_pos = true_pos + true_vel * dt + 0.5 * true_accel * dt**2
    # Sonra hızı güncelle
    true_vel = true_vel + true_accel * dt
    
    # --- B. SENSÖR SİMÜLASYONU (Odometre) ---
    # Senaryo: Odometremiz bize HIZ (Velocity) veriyor.
    # Ama o da mükemmel değil, ölçüm hatası var.
    odo_noise_std = 0.2 # Hız ölçüm hatası (m/s)
    odo_measurement = true_vel + np.random.normal(0, odo_noise_std, 3)
    
    # --- C. KAYIT ---
    log_true_pos.append(true_pos.copy())
    log_true_vel.append(true_vel.copy())
    log_odo_meas.append(odo_measurement)

# Numpy array'e çevir
log_true_pos = np.array(log_true_pos)
log_true_vel = np.array(log_true_vel)
log_odo_meas = np.array(log_odo_meas)

# --- 5. GÖRSELLEŞTİRME ---
t = np.arange(0, total_time, dt)

plt.figure(figsize=(12, 8))

# Grafik 1: 3D Konum (Ne çizdik?)
ax = plt.subplot(2, 2, 1, projection='3d')
ax.plot(log_true_pos[:,0], log_true_pos[:,1], log_true_pos[:,2], 'b-', label='Gerçek Rota')
ax.set_title('3D Hareket (Ground Truth)')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()

# Grafik 2: Hız Profili (X Ekseni)
plt.subplot(2, 2, 2)
plt.plot(t, log_true_vel[:,0], 'g-', label='Gerçek Hız (Vx)')
plt.plot(t, log_odo_meas[:,0], 'r.', alpha=0.3, label='Odometre Ölçümü (Gürültülü)')
plt.title('Hız: Gerçek vs Odometre (X Ekseni)')
plt.xlabel('Zaman (s)')
plt.legend()

# Grafik 3: İvme Gürültüsünün Etkisi (Yol pürüzsüz mü?)
# Hızın nasıl titrediğine bak. İşte o Gaussian process noise.
plt.subplot(2, 2, 3)
plt.plot(t, log_true_vel[:,1], 'k-', label='Gerçek Hız (Vy)')
plt.title('Y Ekseni Hızındaki Titreşim (Process Noise Etkisi)')
plt.xlabel('Zaman (s)')
plt.grid(True)

plt.tight_layout()
plt.show()
