import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. AYARLAR ---
dt = 0.1                # Zaman adımı (saniye)
total_time = 20.0       # Toplam uçuş süresi (saniye)
steps = int(total_time / dt)

# --- 2. BAŞLANGIÇ DURUMU (State Initialization) ---
# Uçak (0,0,0) noktasında, hızı 0.
# Konum: x=0, y=0, z=0
# Hız: vx=0, vy=0, vz=0
true_state = np.zeros(6) 

# --- 3. SABİT İVME (Constant Acceleration) ---
# Uçak ne yapıyor?
# ax = 2.0  -> Doğuya doğru hızlanıyor
# ay = 0.5  -> Hafifçe Kuzeye kayıyor (rüzgar veya dönüş)
# az = 1.0  -> Tırmanıyor
accel_true = np.array([2.0, 0.5, 1.0]) 

# Verileri saklamak için depolar (Log)
ground_truth_log = []

# --- 4. SİMÜLASYON DÖNGÜSÜ (Ground Truth Generator) ---
print(f"Simülasyon Başlıyor: {steps} adım...")

for i in range(steps):
    # a. Konum Güncelleme (p = p + v*dt + 0.5*a*dt^2)
    # true_state[0:3] -> x, y, z
    # true_state[3:6] -> vx, vy, vz
    
    true_state[0:3] = true_state[0:3] + (true_state[3:6] * dt) + (0.5 * accel_true * dt**2)
    
    # b. Hız Güncelleme (v = v + a*dt)
    true_state[3:6] = true_state[3:6] + (accel_true * dt)
    
    # Kayıt et (Kopyasını alarak)
    ground_truth_log.append(true_state.copy())

# Numpy array'e çevir (İşlemesi kolay olsun)
ground_truth_log = np.array(ground_truth_log)

# --- 5. GÖRSELLEŞTİRME (3D Plot) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# X, Y, Z eksenlerini ayır
gt_x = ground_truth_log[:, 0]
gt_y = ground_truth_log[:, 1]
gt_z = ground_truth_log[:, 2]

ax.plot(gt_x, gt_y, gt_z, label='Gerçek Rota (Ground Truth)', color='green', linewidth=2)
ax.set_xlabel('X (Doğu) [m]')
ax.set_ylabel('Y (Kuzey) [m]')
ax.set_zlabel('Z (İrtifa) [m]')
ax.set_title('PART 1: Mükemmel Uçuş Yörüngesi')
ax.legend()
plt.show()

print(f"Son Konum: X={gt_x[-1]:.2f}, Y={gt_y[-1]:.2f}, Z={gt_z[-1]:.2f}")
