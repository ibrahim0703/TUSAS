import numpy as np
import matplotlib.pyplot as plt

# --- YARDIMCI FONKSİYON: Euler'den Quaternion'a ---
def euler_to_quaternion(roll, pitch, yaw):
    """
    Basit trigonometri ile Euler açılarını (Radyan) Quaternion'a (w,x,y,z) çevirir.
    Scipy kütüphanesi kurmana gerek kalmasın diye elle yazdım.
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])

# --- YARDIMCI FONKSİYON: Quaternion ile Vektör Döndürme ---
def rotate_vector(q, v):
    """
    Bir vektörü (v), bir quaternion (q) ile döndürür.
    Formül: v_new = q * v * q_inverse
    """
    w, x, y, z = q
    # Rodrigues formülünün vektörel hali (Hızlı hesap için)
    q_vec = np.array([x, y, z])
    t = 2 * np.cross(q_vec, v)
    return v + w * t + np.cross(q_vec, t)

# ==========================================
# 1. AYARLAR VE GERÇEK YÖRÜNGE (PART 1'den)
# ==========================================
dt = 0.1
total_time = 30.0
t = np.arange(0, total_time, dt)
steps = len(t)

# Yörünge Parametreleri
radius = 500.0
angular_vel = 0.15  # Dönüş hızı (Yaw Rate)
climb_rate = 10.0   # Tırmanma
speed = radius * angular_vel # Yatay Hız (v = r*w) -> 75 m/s

# A. True Position (Nav Frame)
gt_x = radius * np.cos(angular_vel * t)
gt_y = radius * np.sin(angular_vel * t)
gt_z = climb_rate * t + 1000

# B. True Velocity (Nav Frame)
gt_vx = -speed * np.sin(angular_vel * t)
gt_vy =  speed * np.cos(angular_vel * t)
gt_vz = np.full_like(t, climb_rate)

# C. True Acceleration (Nav Frame) - Sadece Merkezkaç var
gt_ax = -(speed**2 / radius) * np.cos(angular_vel * t)
gt_ay = -(speed**2 / radius) * np.sin(angular_vel * t)
gt_az = np.zeros_like(t)

# ==========================================
# 2. SENSÖR VERİSİ ÜRETİMİ (PART 2 BAŞLIYOR)
# ==========================================

# -- Gürültü Ayarları (Standart Sapmalar) --
gps_noise_std = 5.0        # GPS hatası: 5 metre
accel_noise_std = 0.2      # İvmeölçer gürültüsü
gyro_noise_std = 0.01      # Jiroskop gürültüsü
accel_bias = np.array([0.1, -0.1, 0.05]) # İvmeölçer Kayması (Sabit Hata)

# Verileri saklayacağımız listeler
imu_accel_data = [] # İvmeölçer ne okuyacak?
imu_gyro_data = []  # Jiroskop ne okuyacak?
gps_data = []       # GPS ne okuyacak?
true_quats = []     # Gerçek yönelimler (Referans için)

gravity = np.array([0, 0, 9.81]) # Yerçekimi vektörü (Aşağı doğru)

print("Sensör verileri üretiliyor...")

for i in range(steps):
    # --- ADIM A: Gerçek Yönelimi (Orientation) Hesapla ---
    # 1. Yaw (Dönüş): Zamanla değişiyor
    true_yaw = angular_vel * t[i] + (np.pi/2) # Teğet olması için 90 derece ekledik
    
    # 2. Pitch (Tırmanma Açısı): Sabit
    # tan(pitch) = vy / vx -> 10 / 75
    true_pitch = np.arctan(climb_rate / speed)
    
    # 3. Roll (Yatış Açısı): Dönüş yapmak için yatması lazım (Coordinated Turn)
    # tan(roll) = v^2 / (r * g) -> Merkezkaç dengelemek için
    centripetal_accel = (speed**2) / radius
    true_roll = np.arctan(centripetal_accel / 9.81)
    
    # Euler'den Quaternion'a geç
    q_true = euler_to_quaternion(true_roll, true_pitch, true_yaw)
    true_quats.append(q_true)

    # --- ADIM B: IMU Simülasyonu (Accel) ---
    # İvmeölçer "Specific Force" ölçer -> (Gerçek İvme - Yerçekimi)
    # Ama sensörün kendi ekseninde (Body Frame) ölçer.
    
    # 1. Nav Frame'deki Net İvme (Hızlanma - Yerçekimi)
    # Yerçekimi vektörü (0,0, -9.81) değil, (0,0, 9.81) çıkarılır çünkü g aşağı çekerse sensör yukarı basar.
    current_acc_nav = np.array([gt_ax[i], gt_ay[i], gt_az[i]]) + gravity
    
    # 2. Bunu Body Frame'e döndür (Quaternion'un Tersi ile)
    # rotate_vector fonksiyonumuz q ile döndürür, tersine döndürmek için q'nun eşleniğini (-x,-y,-z) kullanırız.
    q_inv = np.array([q_true[0], -q_true[1], -q_true[2], -q_true[3]])
    acc_body = rotate_vector(q_inv, current_acc_nav)
    
    # 3. Gürültü ve Bias ekle
    noise_a = np.random.normal(0, accel_noise_std, 3)
    measured_accel = acc_body + noise_a + accel_bias
    imu_accel_data.append(measured_accel)
    
    # --- ADIM C: IMU Simülasyonu (Gyro) ---
    # Jiroskop Body Frame'deki açısal hızı ölçer.
    # Bu senaryoda dönüş sabit olduğu için Body Frame'de sadece Z ekseni (Yaw) değil, 
    # yatık olduğu için bileşenlerine ayrılır. Basitlik için Nav Frame açısal hızını döndürüyoruz.
    
    nav_angular_vel = np.array([0, 0, angular_vel]) # Sadece Z etrafında dönüyoruz (Nav Frame)
    gyro_body = rotate_vector(q_inv, nav_angular_vel)
    
    noise_g = np.random.normal(0, gyro_noise_std, 3)
    measured_gyro = gyro_body + noise_g
    imu_gyro_data.append(measured_gyro)

    # --- ADIM D: GPS Simülasyonu ---
    # Gerçek Konum + Gürültü
    noise_gps = np.random.normal(0, gps_noise_std, 3)
    measured_gps = np.array([gt_x[i], gt_y[i], gt_z[i]]) + noise_gps
    gps_data.append(measured_gps)

# Numpy formatına çevir
imu_accel_data = np.array(imu_accel_data)
gps_data = np.array(gps_data)

# ==========================================
# 3. GÖRSELLEŞTİRME (Kirli Veriyi Gör)
# ==========================================
plt.figure(figsize=(12, 6))

# GPS vs Gerçek (Sadece X-Y düzlemi)
plt.subplot(1, 2, 1)
plt.plot(gt_x, gt_y, 'g-', linewidth=2, label='Gerçek Rota')
plt.plot(gps_data[:,0], gps_data[:,1], 'r.', markersize=2, label='GPS Ölçümü (Gürültülü)')
plt.title('GPS Gürültüsü (Kuş Bakışı)')
plt.legend()
plt.grid()

# İvmeölçer Verisi (X Ekseni)
plt.subplot(1, 2, 2)
plt.plot(t, imu_accel_data[:,0], 'b-', alpha=0.6, label='İvmeölçer X (Gürültülü + Bias)')
plt.axhline(0, color='k', linestyle='--') # Gerçekte 0'a yakın olmalı (Düz uçuş bileşenleri hariç)
plt.title('IMU İvmeölçer Verisi (Sensör X Ekseni)')
plt.legend()
plt.grid()

plt.show()

print("PART 2 Tamamlandı. Elimizde artık 'Kirli' sensör verileri var.")
