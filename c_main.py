# “””
main.py  —  Stereo Visual-Inertial Odometry Pipeline

Drone / TUM-VI dataset  |  Kalman 9D fused speed estimation

Pipeline:
t0: stereo → 3D noktalar
IMU: t0→t1 arası entegre → Kalman predict
LK flow (forward-backward doğrulamalı) → t0→t1 eşleşme
3D-2D eşleştir (vektörize)
SQPNP → tvec
Kalman update → filtrelenmiş hız

Çıktı: estimated_trajectory.csv
“””

import cv2
import numpy as np
import glob
import os
from c_tracker import StereoOdometryTracker

# ─────────────────────────────────────────────────────────────────────────────

# YARDIMCI FONKSİYONLAR

# ─────────────────────────────────────────────────────────────────────────────

def load_imu_data(imu_csv_path: str) -> dict:
“””
TUM-VI  mav0/imu0/data.csv  →  {timestamp_ns: (gyro(3,), accel(3,))}

```
Format:  timestamp,wx,wy,wz,ax,ay,az
"""
imu_data = {}
if not os.path.exists(imu_csv_path):
    print(f"[UYARI] IMU dosyasi bulunamadi: {imu_csv_path}")
    print("[UYARI] Yalnizca vizyon ile devam ediliyor.")
    return imu_data

with open(imu_csv_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        parts = line.split(',')
        if len(parts) < 7:
            continue
        try:
            ts    = int(parts[0])
            gyro  = np.array([float(parts[1]),
                              float(parts[2]),
                              float(parts[3])])
            accel = np.array([float(parts[4]),
                              float(parts[5]),
                              float(parts[6])])
            imu_data[ts] = (gyro, accel)
        except ValueError:
            continue

print(f"[IMU] {len(imu_data)} ornek yuklendi.")
return imu_data
```

def get_imu_between(imu_data: dict, ts_start: int, ts_end: int) -> list:
“””
ts_start ile ts_end arasındaki IMU örneklerini
zaman sırasına göre döndürür.
“””
samples = [
(ts, g, a)
for ts, (g, a) in imu_data.items()
if ts_start <= ts <= ts_end
]
return sorted(samples, key=lambda x: x[0])

def parse_timestamp(path: str) -> int:
“”“Dosya adından timestamp al: ‘1234567890.png’ → 1234567890”””
return int(os.path.basename(path).split(’.’)[0])

# ─────────────────────────────────────────────────────────────────────────────

# ANA PIPELINE

# ─────────────────────────────────────────────────────────────────────────────

def main():
print(”[SYSTEM] Stereo VIO Pipeline baslatiliyor (SQPNP + Kalman 9D)…”)

```
tracker = StereoOdometryTracker()

# ── Dosya yolları ─────────────────────────────────────────────────────────
left_images  = sorted(glob.glob('mav0/cam0/data/*.png'))
right_images = sorted(glob.glob('mav0/cam1/data/*.png'))
imu_csv      = 'mav0/imu0/data.csv'

if not left_images or not right_images:
    print("[HATA] Goruntu bulunamadi. 'mav0/cam0/data/' yolunu kontrol et.")
    return

n_frames = min(len(left_images), len(right_images))
if len(left_images) != len(right_images):
    print(f"[UYARI] Goruntu sayisi eslesmiyor: "
          f"{len(left_images)} sol / {len(right_images)} sag")
    print(f"[UYARI] {n_frames} kare isleniyor.")

# ── IMU ───────────────────────────────────────────────────────────────────
imu_data = load_imu_data(imu_csv)
use_imu  = len(imu_data) > 0

# ── Log dosyası ───────────────────────────────────────────────────────────
log_path = "estimated_trajectory.csv"
log = open(log_path, "w")
log.write("timestamp_ns,"
          "tx,ty,tz,"
          "speed_raw_m_s,"
          "speed_kalman_m_s,"
          "inliers,"
          "imu_fused\n")

# ── İstatistik sayaçları ──────────────────────────────────────────────────
n_total = n_success = n_skip_pts = n_skip_speed = n_skip_pnp = 0

print(f"[SYSTEM] {n_frames} kare islenecek.")
print("-" * 50)

for i in range(n_frames - 1):
    n_total += 1

    # ── Zaman adımı ───────────────────────────────────────────────────────
    ts_t0 = parse_timestamp(left_images[i])
    ts_t1 = parse_timestamp(left_images[i + 1])
    dt    = (ts_t1 - ts_t0) * 1e-9        # nanosaniye → saniye
    if dt <= 0 or dt > 1.0:               # 1s üzeri boşluk → atla
        continue

    # ── Görüntü yükle ─────────────────────────────────────────────────────
    L0 = cv2.imread(left_images[i],      cv2.IMREAD_GRAYSCALE)
    R0 = cv2.imread(right_images[i],     cv2.IMREAD_GRAYSCALE)
    L1 = cv2.imread(left_images[i + 1],  cv2.IMREAD_GRAYSCALE)
    R1 = cv2.imread(right_images[i + 1], cv2.IMREAD_GRAYSCALE)

    if L0 is None or R0 is None or L1 is None or R1 is None:
        print(f"[UYARI] Frame {i}: goruntu yuklenemedi, atlaniyor.")
        continue

    # ── ADIM 1: t0 stereo → 3D noktalar + rectified sol ──────────────────
    pts_t0, pts_3d_t0, rect_L0 = tracker.process_space_get_depth(L0, R0)

    if len(pts_t0) < 8:
        n_skip_pts += 1
        continue

    # ── ADIM 2: IMU entegrasyonu → Kalman predict ─────────────────────────
    tracker.imu.reset()
    imu_fused = False

    if use_imu:
        samples = get_imu_between(imu_data, ts_t0, ts_t1)
        if len(samples) >= 2:
            for j in range(len(samples) - 1):
                ts_a, g_a, a_a = samples[j]
                ts_b, _,   _   = samples[j + 1]
                sub_dt = (ts_b - ts_a) * 1e-9
                tracker.imu.integrate(g_a, a_a, sub_dt)
            imu_fused = True

    # IMU varsa Kalman'ı öne taşı
    if imu_fused:
        tracker.kalman_predict_from_imu(dt)

    # ── ADIM 3: t1 sol → rectify ──────────────────────────────────────────
    rect_L1 = tracker.rectify_image(L1, side='left')

    # ── ADIM 4: Optical flow (forward-backward doğrulamalı) ───────────────
    p0_arr, p1_arr, _ = tracker.track_time_get_flow(rect_L0, rect_L1, pts_t0)

    if len(p0_arr) < 8:
        n_skip_pts += 1
        continue

    # ── ADIM 5: 3D-2D eşleştir (vektörize) ───────────────────────────────
    matched_3d, matched_2d = tracker.match_3d_2d(
        pts_t0, pts_3d_t0, p0_arr, p1_arr
    )

    if len(matched_3d) < 6:
        n_skip_pts += 1
        continue

    # ── ADIM 6: SQPNP odometry ────────────────────────────────────────────
    rvec, tvec, inliers = tracker.calculate_odometry(
        matched_3d.astype(np.float32),
        matched_2d.astype(np.float32)
    )

    if tvec is None:
        n_skip_pnp += 1
        continue

    # ── ADIM 7: Ham hız (Kalman öncesi) ──────────────────────────────────
    speed_raw = float(np.linalg.norm(tvec)) / dt
    if speed_raw > tracker.SPEED_LIMIT:
        n_skip_speed += 1
        continue

    # ── ADIM 8: Kalman update → filtrelenmiş hız ─────────────────────────
    speed_kalman = tracker.kalman_update_from_pnp(tvec, len(inliers))

    # ── ADIM 9: Log yaz ───────────────────────────────────────────────────
    tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
    log.write(
        f"{ts_t1},"
        f"{tx:.6f},{ty:.6f},{tz:.6f},"
        f"{speed_raw:.4f},"
        f"{speed_kalman:.4f},"
        f"{len(inliers)},"
        f"{int(imu_fused)}\n"
    )

    n_success += 1

    # ── İlerleme göster ───────────────────────────────────────────────────
    if n_total % 500 == 0:
        print(f"  [{n_total}/{n_frames}]  "
              f"basari={100*n_success/n_total:.1f}%  "
              f"kalman_hiz={speed_kalman:.3f} m/s")

log.close()

# ── Özet ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  Toplam kare         : {n_total}")
print(f"  Basarili tahmin     : {n_success}  "
      f"({100*n_success/max(n_total,1):.1f}%)")
print(f"  Nokta yetersiz      : {n_skip_pts}")
print(f"  PnP basarisiz       : {n_skip_pnp}")
print(f"  Hiz filtresi (>{tracker.SPEED_LIMIT}m/s) : {n_skip_speed}")
print("=" * 55)
print(f"  Cikti: {log_path}")
print("=" * 55)

# ── Kalman son durum ──────────────────────────────────────────────────────
print("\n[KALMAN] Son durum:")
print(f"  Hiz        : {tracker.kalman.get_speed():.4f} m/s")
print(f"  Hiz vektoru: {tracker.kalman.get_velocity()}")
print(f"  Gyro bias  : {tracker.kalman.get_gyro_bias()}")
```

if **name** == “**main**”:
main()