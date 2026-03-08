# -*- coding: utf-8 -*-
"""
main.py  —  TUM-VI Stereo Visual-Inertial Odometry Pipeline (Optimized)
=========================================================================
Drone anlık hız tahmini: Stereo görü + IMU + 9D Kalman filtresi

Optimizasyonlar:
  - Düşük inlier frame rejection → sadece predict ile devam
  - Hız sürekliliği kontrolü → ani spike'ları filtrele
  - dt-adaptif işleme → frame drop durumunda güvenilirlik düşür
  - PnP başarısız → predict-only fallback

Beklenen dizin yapısı (TUM-VI formatı):
    mav0/
        cam0/data/*.png   (sol kamera)
        cam1/data/*.png   (sağ kamera)
        imu0/data.csv     (IMU: timestamp, gx, gy, gz, ax, ay, az)

Çıktı:
    estimated_trajectory.csv
        timestamp, vx, vy, vz, speed_m_s, inliers, imu_fused
"""

import cv2
import numpy as np
import glob
import os
from tracker import StereoOdometryTracker


# ─────────────────────────────────────────────────────────────────────────────
# IMU YÜKLEME
# ─────────────────────────────────────────────────────────────────────────────
def load_imu_data(imu_csv_path: str) -> dict:
    imu_data = {}
    if not os.path.exists(imu_csv_path):
        print(f"[UYARI] IMU dosyası bulunamadı: {imu_csv_path}")
        print("[UYARI] Yalnızca görü ile devam ediliyor.")
        return imu_data

    with open(imu_csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            ts    = int(parts[0])
            gyro  = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            accel = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
            imu_data[ts] = (gyro, accel)

    print(f"[IMU] {len(imu_data)} örnek yüklendi.")
    return imu_data


def get_imu_between(imu_data: dict, ts_start: int, ts_end: int) -> list:
    samples = [
        (ts, g, a)
        for ts, (g, a) in imu_data.items()
        if ts_start <= ts <= ts_end
    ]
    return sorted(samples, key=lambda x: x[0])


# ─────────────────────────────────────────────────────────────────────────────
# ANA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  TUM-VI Stereo VIO Pipeline (Optimized)")
    print("  Stereo + IMU + 9D Kalman + Innovation Gate")
    print("=" * 65)

    tracker = StereoOdometryTracker()

    # ── Dosya yolları ─────────────────────────────────────────────────────────
    left_images  = sorted(glob.glob('mav0/cam0/data/*.png'))
    right_images = sorted(glob.glob('mav0/cam1/data/*.png'))
    imu_csv      = 'mav0/imu0/data.csv'

    # ── Ön kontroller ─────────────────────────────────────────────────────────
    if not left_images or not right_images:
        print("[HATA] Görüntü bulunamadı. Dizin yolunu kontrol et.")
        return

    n_frames = min(len(left_images), len(right_images))
    if len(left_images) != len(right_images):
        print(f"[UYARI] Sol/sağ görüntü sayısı eşleşmedi: "
              f"{len(left_images)} vs {len(right_images)} — min({n_frames}) kullanılıyor.")

    imu_data = load_imu_data(imu_csv)
    use_imu  = len(imu_data) > 0

    # ── CSV çıktısı ───────────────────────────────────────────────────────────
    log = open("estimated_trajectory.csv", "w")
    log.write("timestamp,vx,vy,vz,speed_m_s,inliers,imu_fused\n")

    print(f"[SYSTEM] {n_frames} kare işlenecek.")
    print(f"[SYSTEM] Optimizasyonlar: GFTT+ORB, FB-Flow, IMU-Rot, InnovGate")

    # ── Sayaçlar ──────────────────────────────────────────────────────────────
    n_total = n_success = n_skip_pts = n_skip_pnp = n_skip_speed = 0
    n_predict_only = n_innov_rejected = 0
    n_abs_thresh = n_ratio_thresh = n_dir_thresh = 0

    for i in range(n_frames - 1):
        n_total += 1

        # ── Zaman adımı ───────────────────────────────────────────────────────
        ts_t0 = int(os.path.basename(left_images[i]).split('.')[0])
        ts_t1 = int(os.path.basename(left_images[i + 1]).split('.')[0])
        dt    = (ts_t1 - ts_t0) * 1e-9

        if dt <= 0:
            continue

        # ── Görüntü yükle ─────────────────────────────────────────────────────
        L0 = cv2.imread(left_images[i],       cv2.IMREAD_GRAYSCALE)
        R0 = cv2.imread(right_images[i],      cv2.IMREAD_GRAYSCALE)
        L1 = cv2.imread(left_images[i + 1],   cv2.IMREAD_GRAYSCALE)
        R1 = cv2.imread(right_images[i + 1],  cv2.IMREAD_GRAYSCALE)

        if any(img is None for img in [L0, R0, L1, R1]):
            continue

        # ─────────────────────────────────────────────────────────────────────
        # ADIM 1: t0 stereo → Hibrit features + 3D noktalar
        # ─────────────────────────────────────────────────────────────────────
        pts_t0, pts_3d_t0, rect_L0 = tracker.process_space_get_depth(L0, R0)

        if len(pts_t0) < 10:    # 6 → 10
            n_skip_pts += 1
            # PnP yok ama Kalman predict ile devam
            tracker.kalman_predict_only(dt)
            # Yine de hız yaz (predict-based)
            speed = tracker.speed_ms
            vx, vy, vz = tracker.velocity_ms
            n_predict_only += 1
            n_success += 1
            log.write(
                f"{ts_t1},"
                f"{vx:.6f},{vy:.6f},{vz:.6f},"
                f"{speed:.4f},"
                f"0,"
                f"{int(use_imu)}\n"
            )
            continue

        # ─────────────────────────────────────────────────────────────────────
        # ADIM 2: IMU preintegration (ts_t0 → ts_t1)
        # ─────────────────────────────────────────────────────────────────────
        tracker.imu.reset()
        imu_fused = False

        if use_imu:
            samples = get_imu_between(imu_data, ts_t0, ts_t1)
            if len(samples) >= 2:
                for j in range(len(samples) - 1):
                    ts_a, g_a, a_a = samples[j]
                    ts_b, _,   _   = samples[j + 1]
                    sub_dt = (ts_b - ts_a) * 1e-9
                    if sub_dt > 0:
                        tracker.imu.integrate(g_a, a_a, sub_dt)
                imu_fused = True

        # ─────────────────────────────────────────────────────────────────────
        # ADIM 3: t1 sol görüntüyü rectify et
        # ─────────────────────────────────────────────────────────────────────
        rect_L1 = tracker.rectify_image(L1, side='left')

        # ─────────────────────────────────────────────────────────────────────
        # ADIM 4: Forward-Backward Optical Flow (IMU-aided)
        # ─────────────────────────────────────────────────────────────────────
        p0_arr, p1_arr, _ = tracker.track_time_get_flow(rect_L0, rect_L1, pts_t0)

        if len(p0_arr) < 10:   # 6 → 10
            n_skip_pts += 1
            tracker.kalman_predict_only(dt)
            speed = tracker.speed_ms
            vx, vy, vz = tracker.velocity_ms
            n_predict_only += 1
            n_success += 1
            log.write(
                f"{ts_t1},"
                f"{vx:.6f},{vy:.6f},{vz:.6f},"
                f"{speed:.4f},"
                f"0,"
                f"{int(imu_fused)}\n"
            )
            continue

        # ─────────────────────────────────────────────────────────────────────
        # ADIM 5: 3D-2D eşleştirme
        # ─────────────────────────────────────────────────────────────────────
        matched_3d, matched_2d = tracker.match_3d_2d(
            pts_t0, pts_3d_t0, p0_arr, p1_arr, dist_thresh=2.0  # 1.5 → 2.0
        )

        if len(matched_3d) < 10:  # 6 → 10
            n_skip_pts += 1
            tracker.kalman_predict_only(dt)
            speed = tracker.speed_ms
            vx, vy, vz = tracker.velocity_ms
            n_predict_only += 1
            n_success += 1
            log.write(
                f"{ts_t1},"
                f"{vx:.6f},{vy:.6f},{vz:.6f},"
                f"{speed:.4f},"
                f"0,"
                f"{int(imu_fused)}\n"
            )
            continue

        # ─────────────────────────────────────────────────────────────────────
        # ADIM 6: PnP odometry (SQPNP + RANSAC)
        # ─────────────────────────────────────────────────────────────────────
        rvec, tvec, inliers = tracker.calculate_odometry(
            matched_3d.astype(np.float32),
            matched_2d.astype(np.float32)
        )

        if tvec is None:
            n_skip_pnp += 1
            # PnP başarısız → predict-only fallback
            tracker.kalman_predict_only(dt)
            speed = tracker.speed_ms
            vx, vy, vz = tracker.velocity_ms
            n_predict_only += 1
            n_success += 1
            log.write(
                f"{ts_t1},"
                f"{vx:.6f},{vy:.6f},{vz:.6f},"
                f"{speed:.4f},"
                f"0,"
                f"{int(imu_fused)}\n"
            )
            continue

        n_inliers = len(inliers)

        # ─────────────────────────────────────────────────────────────────────
        # ADIM 7: Kalman filtresi — IMU predict + PnP update
        #         (düşük inlier otomatik olarak skip edilir)
        #         (innovation gate Kalman içinde)
        # ─────────────────────────────────────────────────────────────────────
        step_result = tracker.kalman_step(tvec.ravel(), dt, n_inliers)

        # Pre-filter / innovation gate istatistikleri
        if step_result == "innov_rejected":
            n_innov_rejected += 1
        elif step_result == "abs_thresh":
            n_abs_thresh += 1
        elif step_result == "ratio_thresh":
            n_ratio_thresh += 1
        elif step_result == "direction_thresh":
            n_dir_thresh += 1

        # ─────────────────────────────────────────────────────────────────────
        # ADIM 8: Hız kontrolü
        # ─────────────────────────────────────────────────────────────────────
        speed = tracker.speed_ms

        # Fiziksel sınır
        MAX_SPEED_MS = 5.0    # TUM-VI Room1 GT max ~2 m/s, güvenlik payı dahil
        if speed > MAX_SPEED_MS:
            # Hızı sınırla — Kalman state'ini clamp'le
            tracker.kalman.x[3:6] *= (MAX_SPEED_MS / speed) * 0.5
            n_skip_speed += 1
            continue

        # Hız sürekliliği kontrolü (sıkılaştırılmış)
        if not tracker.check_speed_continuity(speed, max_ratio=3.0):
            n_skip_speed += 1
            continue

        # ─────────────────────────────────────────────────────────────────────
        # ADIM 9: CSV'e yaz
        # ─────────────────────────────────────────────────────────────────────
        vx, vy, vz = tracker.velocity_ms
        n_success += 1

        log.write(
            f"{ts_t1},"
            f"{vx:.6f},{vy:.6f},{vz:.6f},"
            f"{speed:.4f},"
            f"{n_inliers},"
            f"{int(imu_fused)}\n"
        )

        # İlerleme çıktısı (her 500 frame'de bir)
        if n_success % 500 == 0:
            gate_status = "PASS" if tracker.kalman.last_update_accepted else "REJECT"
            print(f"  [Frame {i:6d}] hız={speed:.3f} m/s | "
                  f"inlier={n_inliers} | imu={'OK' if imu_fused else '--'} | "
                  f"gate={gate_status}")

    log.close()

    # ── Özet ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Toplam kare           : {n_total}")
    print(f"  Başarılı tahmin       : {n_success}")
    print(f"    - PnP ile           : {n_success - n_predict_only}")
    print(f"    - Predict-only      : {n_predict_only}")
    print(f"  Nokta yetersiz        : {n_skip_pts}")
    print(f"  PnP başarısız         : {n_skip_pnp}")
    print(f"  Hız filtresi          : {n_skip_speed}")
    print(f"  Innovation rejected   : {n_innov_rejected}")
    print(f"  Pre-filter rejected   : abs={n_abs_thresh} ratio={n_ratio_thresh} dir={n_dir_thresh}")
    print(f"  Başarı oranı          : {100 * n_success / max(n_total, 1):.1f}%")
    print("=" * 65)
    print("  Çıktı: estimated_trajectory.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()