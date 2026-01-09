import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. SİMÜLASYON AYARLARI ---
dt = 0.1                  # Zaman adımı (10 Hz)
total_time = 30.0         # 30 saniyelik uçuş
t = np.arange(0, total_time, dt)
n_steps = len(t)

# --- 2. FİZİKSEL YÖRÜNGE OLUŞTURMA (GROUND TRUTH) ---
# Senaryo: Uçak 100m/s hızla gidiyor ve sürekli sola dönerek tırmanıyor (Spiral)
radius = 500.0            # Dönüş yarıçapı (metre)
angular_vel = 0.15        # Dönüş hızı (rad/s) -> Yaklaşık 8 derece/sn
climb_rate = 10.0         # Tırmanma hızı (m/s)

# A. KONUM (Position)
# x = r * cos(w*t)
# y = r * sin(w*t)
# z = vz * t
gt_x = radius * np.cos(angular_vel * t)
gt_y = radius * np.sin(angular_vel * t)
gt_z = climb_rate * t + 1000 # 1000 metreden başlasın

# B. HIZ (Velocity - Konumun Türevi)
# vx = -r * w * sin(w*t)
# vy =  r * w * cos(w*t)
# vz = sabit
gt_vx = -radius * angular_vel * np.sin(angular_vel * t)
gt_vy =  radius * angular_vel * np.cos(angular_vel * t)
gt_vz = np.full_like(t, climb_rate) # Sabit tırmanma hızı

# C. İVME (Acceleration - Hızın Türevi) - İleride İvmeölçer (INS) için lazım olacak!
# Merkezkaç ivmesi işin içine giriyor
gt_ax = -radius * (angular_vel**2) * np.cos(angular_vel * t)
gt_ay = -radius * (angular_vel**2) * np.sin(angular_vel * t)
gt_az = np.zeros_like(t) # Dikey ivme yok (sabit hızla tırmanıyor)

# Verileri paketleyelim (İleride kullanmak için)
ground_truth = np.stack([gt_x, gt_y, gt_z, gt_vx, gt_vy, gt_vz], axis=1)

# --- 3. GÖRSELLEŞTİRME ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Gerçek Rotayı Çiz
ax.plot(gt_x, gt_y, gt_z, label='Ground Truth (Gerçek Rota)', color='blue', linewidth=2)

# Başlangıç ve Bitiş Noktalarını İşaretle
ax.scatter(gt_x[0], gt_y[0], gt_z[0], color='green', s=100, label='Başlangıç')
ax.scatter(gt_x[-1], gt_y[-1], gt_z[-1], color='red', s=100, label='Bitiş')

ax.set_title('PART 1: 3D Referans Yörünge (Spiral Tırmanış)')
ax.set_xlabel('X (Doğu) [m]')
ax.set_ylabel('Y (Kuzey) [m]')
ax.set_zlabel('Z (İrtifa) [m]')
ax.legend()
ax.grid(True)

plt.show()

print(f"Veri Üretildi. Toplam Adım: {n_steps}")
print(f"Son Konum: X={gt_x[-1]:.1f}, Y={gt_y[-1]:.1f}, Z={gt_z[-1]:.1f}")
print(f"Bu veriler bizim 'Cevap Anahtarımız' olacak.")
