# -*- coding: utf-8 -*-
"""
main_eskf.py  —  TUM-VI Stereo VIO Pipeline (V2: 15D ESKF)
============================================================
15D Error-State Kalman Filter ile drone anlik hiz tahmini.

V2 Degisiklikler:
  - 15D ESKF (quaternion oryantasyon + accel bias + gyro bias)
  - IMU per-sample predict (her IMU orneginde nominal state propagation)
  - Gravity alignment (ilk IMU verilerinden oryantasyon tahmini)
  - Asama 2: Direct 6DOF Pose Update (v=tvec/dt KALDIRILDI)
  - Asama 3: Continuous Feature Tracking (track_id + sliding window)

Beklenen dizin yapisi (TUM-VI formati):
    mav0/
        cam0/data/*.png   (sol kamera)
        cam1/data/*.png   (sag kamera)
        imu0/data.csv     (IMU: timestamp, gx, gy, gz, ax, ay, az)

Cikti:
    estimated_trajectory.csv
        timestamp, vx, vy, vz, speed_m_s, inliers, imu_fused
"""

import cv2
import numpy as np
import glob
import os
from tracker_eskf import StereoOdometryTracker


# -----------------------------------------------------------------------------
# IMU YUKLEME
# -----------------------------------------------------------------------------
def load_imu_data(imu_csv_path: str) -> dict:
    imu_data = {}
    if not os.path.exists(imu_csv_path):
        print(f"[UYARI] IMU dosyasi bulunamadi: {imu_csv_path}")
        print("[UYARI] Yalnizca goru ile devam ediliyor.")
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

    print(f"[IMU] {len(imu_data)} ornek yuklendi.")
    return imu_data


def get_imu_between(imu_data: dict, ts_start: int, ts_end: int) -> list:
    samples = [
        (ts, g, a)
        for ts, (g, a) in imu_data.items()
        if ts_start <= ts <= ts_end
    ]
    return sorted(samples, key=lambda x: x[0])


# -----------------------------------------------------------------------------
# ANA PIPELINE
# -----------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  TUM-VI Stereo VIO Pipeline (V2: 15D ESKF)")
    print("  Asama 2: Direct 6DOF Pose Update")
    print("  Asama 3: Continuous Feature Tracking")
    print("=" * 65)

    # -- Dosya yollari --------------------------------------------------------
    left_images  = sorted(glob.glob('mav0/cam0/data/*.png'))
    right_images = sorted(glob.glob('mav0/cam1/data/*.png'))
    imu_csv      = 'mav0/imu0/data.csv'

    # -- On kontroller --------------------------------------------------------
    if not left_images or not right_images:
        print("[HATA] Goruntu bulunamadi. Dizin yolunu kontrol et.")
        return

    n_frames = min(len(left_images), len(right_images))
    if len(left_images) != len(right_images):
        print(f"[UYARI] Sol/sag goruntu sayisi eslesmedi: "
              f"{len(left_images)} vs {len(right_images)} — min({n_frames}) kullaniliyor.")

    imu_data = load_imu_data(imu_csv)
    use_imu  = len(imu_data) > 0

    tracker = StereoOdometryTracker()

    # -- Gravity alignment: ilk IMU verilerinden oryantasyon tahmini ----------
    if use_imu:
        all_imu_sorted = sorted(imu_data.items(), key=lambda x: x[0])
        n_init = min(len(all_imu_sorted), 200)
        init_accels = [a for _, (g, a) in all_imu_sorted[:n_init]]

        if len(init_accels) >= 3:
            mean_accel = np.mean(init_accels, axis=0)
            tracker.kalman.initialize_from_gravity(mean_accel)
            q = tracker.kalman.quaternion
            print(f"[ESKF] Gravity alignment: a_mean = "
                  f"[{mean_accel[0]:.4f}, {mean_accel[1]:.4f}, {mean_accel[2]:.4f}]")
            print(f"[ESKF] Initial quaternion: "
                  f"[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
        else:
            print("[ESKF] Gravity alignment icin yeterli IMU verisi yok, "
                  "identity quaternion kullaniliyor.")

    # -- CSV ciktisi ----------------------------------------------------------
    log = open("estimated_trajectory.csv", "w")
    log.write("timestamp,vx,vy,vz,speed_m_s,inliers,imu_fused\n")

    print(f"[SYSTEM] {n_frames} kare islenecek.")
    print(f"[SYSTEM] Asama 2: 6DOF Pose Update (v=tvec/dt kaldirildi)")
    print(f"[SYSTEM] Asama 3: Continuous Tracking (MIN_TRACKED={tracker.MIN_TRACKED_FEATURES})")

    # -- Sayaclar -------------------------------------------------------------
    n_total = n_success = n_skip_pts = n_skip_pnp = n_skip_speed = 0
    n_predict_only = n_innov_rejected = 0
    n_abs_thresh = n_ratio_thresh = n_dir_thresh = 0
    n_new_detect = 0   # Asama 3: yeni feature detection sayaci

    prev_ts = None

    # =========================================================================
    # ANA DONGU — her frame tek tek islenir (continuous tracking)
    # =========================================================================
    for i in range(n_frames):

        # -- Goruntu yukle ----------------------------------------------------
        L = cv2.imread(left_images[i],  cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)

        if L is None or R is None:
            continue

        ts = int(os.path.basename(left_images[i]).split('.')[0])

        # -- Stereo rectify + disparity ---------------------------------------
        # cv2.StereoSGBM_create disparity hesaplar
        rect_L, rect_R, disp = tracker.rectify_stereo(L, R)

        # =====================================================================
        # ILK FRAME: feature detect + sakla, donguye devam
        # =====================================================================
        if i == 0:
            tracker.update_tracked_state(
                np.empty((0, 2), dtype=np.float32),
                np.array([], dtype=np.int64),
                rect_L, disp
            )
            prev_ts = ts
            n_new_detect += 1
            continue

        n_total += 1
        dt = (ts - prev_ts) * 1e-9

        if dt <= 0:
            prev_ts = ts
            continue

        # =====================================================================
        # PREDICT ONCESI STATE KAYDET (Asama 2: pose update icin referans)
        # =====================================================================
        p_prev = tracker.kalman.p.copy()
        q_prev = tracker.kalman.q.copy()

        # =====================================================================
        # IMU PER-SAMPLE PREDICT
        # =====================================================================
        tracker.imu.reset()
        imu_fused = False

        if use_imu:
            tracker.imu.set_bias(tracker.kalman.gyro_bias)
            samples = get_imu_between(imu_data, prev_ts, ts)

            if len(samples) >= 2:
                for j in range(len(samples) - 1):
                    ts_a, g_a, a_a = samples[j]
                    ts_b, _,   _   = samples[j + 1]
                    sub_dt = (ts_b - ts_a) * 1e-9
                    if sub_dt > 0:
                        tracker.kalman.predict(g_a, a_a, sub_dt)
                        tracker.imu.integrate(g_a, a_a, sub_dt)
                imu_fused = True

        if not imu_fused:
            tracker.kalman.predict_no_imu(dt)

        # =====================================================================
        # ASAMA 3: CONTINUOUS TRACKING — onceki frame'den yeni frame'e track
        # =====================================================================
        tracked_3d, tracked_2d, survived_ids = tracker.track_features(rect_L)

        n_tracked  = len(tracked_2d)
        n_inliers  = 0
        step_result = "no_features"
        pnp_ok      = False

        if n_tracked >= 10:
            # =================================================================
            # PnP ODOMETRY (3D onceki frame, 2D guncel frame)
            # =================================================================
            rvec, tvec, inliers = tracker.calculate_odometry(
                tracked_3d.astype(np.float32),
                tracked_2d.astype(np.float32)
            )

            if tvec is not None:
                n_inliers = len(inliers)
                pnp_ok = True

                # =============================================================
                # ASAMA 2: 6DOF KALMAN UPDATE (v=tvec/dt YOK)
                # =============================================================
                step_result = tracker.kalman_update_step(
                    rvec, tvec, dt, n_inliers, p_prev, q_prev
                )

                # Sadece inlier feature'lari sonraki frame'e tasi
                inlier_idx  = inliers.ravel()
                inlier_2d   = tracked_2d[inlier_idx]
                inlier_ids  = survived_ids[inlier_idx]

                # State guncelle: re-triangulate + gerekirse yeni detect
                n_before = len(inlier_2d)
                tracker.update_tracked_state(
                    inlier_2d, inlier_ids, rect_L, disp
                )
                n_after = (len(tracker._tracked_pts_2d)
                           if tracker._tracked_pts_2d is not None else 0)
                if n_after > n_before:
                    n_new_detect += 1

            else:
                n_skip_pnp += 1
                # PnP basarisiz: tum tracked feature'lari koru
                tracker.update_tracked_state(
                    tracked_2d, survived_ids, rect_L, disp
                )
        else:
            n_skip_pts += 1
            # Yetersiz feature: sifirdan detect
            tracker.update_tracked_state(
                np.empty((0, 2), dtype=np.float32),
                np.array([], dtype=np.int64),
                rect_L, disp
            )
            n_new_detect += 1

        # -- Pre-filter / innovation gate istatistikleri ----------------------
        if step_result == "innov_rejected":
            n_innov_rejected += 1
        elif step_result == "abs_thresh":
            n_abs_thresh += 1
        elif step_result == "ratio_thresh":
            n_ratio_thresh += 1
        elif step_result == "direction_thresh":
            n_dir_thresh += 1

        # =====================================================================
        # HIZ KONTROLU
        # =====================================================================
        speed = tracker.speed_ms

        MAX_SPEED_MS = 5.0
        if speed > MAX_SPEED_MS:
            tracker.kalman.v *= (MAX_SPEED_MS / speed)
            n_skip_speed += 1
            prev_ts = ts
            continue

        if not tracker.check_speed_continuity(speed, max_ratio=3.0):
            n_skip_speed += 1
            prev_ts = ts
            continue

        # =====================================================================
        # CSV'E YAZ
        # =====================================================================
        vx, vy, vz = tracker.velocity_ms
        n_success += 1

        if not pnp_ok:
            n_predict_only += 1

        log.write(
            f"{ts},"
            f"{vx:.6f},{vy:.6f},{vz:.6f},"
            f"{speed:.4f},"
            f"{n_inliers},"
            f"{int(imu_fused)}\n"
        )

        # Ilerleme ciktisi (her 500 frame'de bir)
        if n_success % 500 == 0:
            gate_status = "PASS" if tracker.kalman.last_update_accepted else "REJECT"
            n_trk = (len(tracker._tracked_pts_2d)
                     if tracker._tracked_pts_2d is not None else 0)
            print(f"  [Frame {i:6d}] hiz={speed:.3f} m/s | "
                  f"inlier={n_inliers} | tracked={n_trk} | "
                  f"imu={'OK' if imu_fused else '--'} | "
                  f"gate={gate_status}")

        prev_ts = ts

    log.close()

    # -- Ozet -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"  Toplam kare           : {n_total}")
    print(f"  Basarili tahmin       : {n_success}")
    print(f"    - PnP ile           : {n_success - n_predict_only}")
    print(f"    - Predict-only      : {n_predict_only}")
    print(f"  Nokta yetersiz        : {n_skip_pts}")
    print(f"  PnP basarisiz         : {n_skip_pnp}")
    print(f"  Hiz filtresi          : {n_skip_speed}")
    print(f"  Innovation rejected   : {n_innov_rejected}")
    print(f"  Pre-filter rejected   : abs={n_abs_thresh} ratio={n_ratio_thresh} dir={n_dir_thresh}")
    print(f"  Yeni feature detect   : {n_new_detect}")
    print(f"  Basari orani          : {100 * n_success / max(n_total, 1):.1f}%")
    print("=" * 65)
    print("  Cikti: estimated_trajectory.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
