import cv2
import numpy as np
import glob
import os
from c_tracker import StereoOdometryTracker


def load_imu_data(imu_csv_path):
    """TUM-VI imu0/data.csv → dict {timestamp_ns: (gyro, accel)}"""
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
            ts    = int(parts[0])
            gyro  = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            accel = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
            imu_data[ts] = (gyro, accel)
    print(f"[IMU] {len(imu_data)} ornek yuklendi.")
    return imu_data


def get_imu_between(imu_data, ts_start, ts_end):
    samples = [
        (ts, g, a) for ts, (g, a) in imu_data.items()
        if ts_start <= ts <= ts_end
    ]
    return sorted(samples, key=lambda x: x[0])


def main():
    print("[SYSTEM] VIO Pipeline baslatiliyor (Stereo + IMU)...")
    tracker = StereoOdometryTracker()

    # --- Dosya yollarini kendi sistemine gore ayarla ---
    left_images  = sorted(glob.glob('mav0/cam0/data/*.png'))
    right_images = sorted(glob.glob('mav0/cam1/data/*.png'))
    imu_csv      = 'mav0/imu0/data.csv'

    if not left_images or not right_images:
        print("[HATA] Goruntu bulunamadi.")
        return

    if len(left_images) != len(right_images):
        print(f"[UYARI] Sol/sag goruntu sayisi eslesmedi: {len(left_images)} vs {len(right_images)}")

    imu_data = load_imu_data(imu_csv)
    use_imu  = len(imu_data) > 0

    log = open("estimated_trajectory.csv", "w")
    log.write("timestamp,tx,ty,tz,speed_m_s,inliers,imu_fused\n")

    print(f"[SYSTEM] {len(left_images)} kare islenecek. Cikmak icin 'q'.")

    n_total = n_success = n_skip_pts = n_skip_speed = 0

    for i in range(len(left_images) - 1):
        n_total += 1

        # --- Zaman adimi ---
        ts_t0 = int(os.path.basename(left_images[i]).split('.')[0])
        ts_t1 = int(os.path.basename(left_images[i+1]).split('.')[0])
        dt    = (ts_t1 - ts_t0) * 1e-9
        if dt <= 0:
            continue

        # --- Goruntu yukle ---
        L0 = cv2.imread(left_images[i],     cv2.IMREAD_GRAYSCALE)
        R0 = cv2.imread(right_images[i],    cv2.IMREAD_GRAYSCALE)
        L1 = cv2.imread(left_images[i+1],   cv2.IMREAD_GRAYSCALE)
        R1 = cv2.imread(right_images[i+1],  cv2.IMREAD_GRAYSCALE)

        if L0 is None or R0 is None or L1 is None or R1 is None:
            print(f"[UYARI] Frame {i}: goruntu yuklenemedi, atlaniyor.")
            continue

        # --- ADIM 1: t0 stereo → derinlik + rectified noktalar ---
        # process_space_get_depth her zaman 3 deger dondurur:
        #   pts_rect (N,1,2) | pts_3d (N,3) | rect_l (goruntu)
        pts_t0, pts_3d_t0, rect_L0 = tracker.process_space_get_depth(L0, R0)

        if len(pts_t0) < 6:
            n_skip_pts += 1
            continue

        # --- IMU entegrasyonu ---
        imu_fused = False
        if use_imu:
            tracker.imu.reset()
            samples = get_imu_between(imu_data, ts_t0, ts_t1)
            if len(samples) >= 2:
                for j in range(len(samples) - 1):
                    ts_a, g_a, a_a = samples[j]
                    ts_b, _,   _   = samples[j+1]
                    tracker.imu.integrate(g_a, a_a, (ts_b - ts_a) * 1e-9)
                imu_fused = True

        # --- ADIM 2: t1 sol goruntusunu rectify et ---
        rect_L1 = tracker.rectify_image(L1, side='left')

        # --- ADIM 3: Optical flow (rectified uzayda) ---
        p0_list, p1_list, _ = tracker.track_time_get_flow(rect_L0, rect_L1, pts_t0)

        if len(p0_list) < 6:
            n_skip_pts += 1
            continue

        # --- ADIM 4: 3D-2D nokta eslestir ---
        # pts_t0 ile p0_list ayni sirayla uretildi (LK flow girdi noktalarini
        # aynen korur, sadece st==1 olanlar kalir)
        # Eslestirme: p0_list icindeki her nokta icin pts_t0'daki en yakin noktayi bul
        p0_arr  = np.array(p0_list,  dtype=np.float32)
        pts_arr = pts_t0.reshape(-1, 2)

        matched_3d, matched_2d = [], []
        for k in range(len(p0_arr)):
            dists = np.linalg.norm(pts_arr - p0_arr[k], axis=1)
            idx   = int(np.argmin(dists))
            if dists[idx] < 1.5:
                matched_3d.append(pts_3d_t0[idx])
                matched_2d.append(p1_list[k])

        if len(matched_3d) < 6:
            n_skip_pts += 1
            continue

        # --- ADIM 5: PnP odometry ---
        rvec, tvec, inliers = tracker.calculate_odometry(
            np.array(matched_3d, dtype=np.float32),
            np.array(matched_2d, dtype=np.float32)
        )

        if tvec is None:
            continue

        # --- ADIM 6: IMU fuzyon ---
        _, _, imu_dp = tracker.imu.get_prediction()
        if imu_fused:
            tvec = tracker.fuse_imu_vision(tvec, imu_dp, len(inliers))

        speed = float(np.linalg.norm(tvec)) / dt
        if speed > 4.0:
            n_skip_speed += 1
            continue

        n_success += 1
        tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
        log.write(f"{ts_t1},{tx:.6f},{ty:.6f},{tz:.6f},{speed:.4f},{len(inliers)},{int(imu_fused)}\n")
        
    print("\n" + "="*50)
    print(f"  Toplam kare       : {n_total}")
    print(f"  Basarili tahmin   : {n_success}")
    print(f"  Nokta yetersiz    : {n_skip_pts}")
    print(f"  Hiz filtresi      : {n_skip_speed}")
    print(f"  Basari orani      : {100*n_success/max(n_total,1):.1f}%")
    print("="*50)
    print("  'estimated_trajectory.csv' olusturuldu.")
    print("="*50)


if __name__ == "__main__":
    main()
