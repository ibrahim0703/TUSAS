import cv2
import numpy as np
import glob
import os
from tracker import StereoOdometryTracker


# ─────────────────────────────────────────────────────────────────────────────
# IMU veri okuyucu
# ─────────────────────────────────────────────────────────────────────────────
def load_imu_data(imu_csv_path):
    """
    TUM-VI imu0/data.csv formatını yükle.
    Kolon sırası: timestamp[ns], wx, wy, wz, ax, ay, az
    Döndürür: dict {timestamp_ns: (gyro[3], accel[3])}
    """
    imu_data = {}
    if not os.path.exists(imu_csv_path):
        print(f"[UYARI] IMU dosyası bulunamadı: {imu_csv_path}")
        print("[UYARI] Yalnızca vizyon tabanlı devam ediliyor.")
        return imu_data

    with open(imu_csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            ts   = int(parts[0])
            gyro  = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            accel = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
            imu_data[ts] = (gyro, accel)

    print(f"[IMU] {len(imu_data)} örnek yüklendi.")
    return imu_data


def get_imu_between(imu_data, ts_start_ns, ts_end_ns):
    """ts_start ve ts_end arasındaki IMU örneklerini sıralı döndür."""
    samples = [
        (ts, gyro, accel)
        for ts, (gyro, accel) in imu_data.items()
        if ts_start_ns <= ts <= ts_end_ns
    ]
    return sorted(samples, key=lambda x: x[0])


# ─────────────────────────────────────────────────────────────────────────────
# Ana döngü
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[SYSTEM] VIO Pipeline başlatılıyor (Stereo + IMU)...")
    tracker = StereoOdometryTracker()

    # ── Dosya yolları (kendi sistemine göre düzenle) ──────────────────────
    left_images  = sorted(glob.glob('mav0/cam0/data/*.png'))
    right_images = sorted(glob.glob('mav0/cam1/data/*.png'))
    imu_csv      = 'mav0/imu0/data.csv'     # TUM-VI standart yolu

    if not left_images or not right_images:
        print("[HATA] Görüntü bulunamadı. Dosya yollarını kontrol et.")
        return

    imu_data = load_imu_data(imu_csv)
    use_imu  = len(imu_data) > 0

    log_file = open("estimated_trajectory.csv", "w")
    log_file.write("timestamp,tx,ty,tz,speed_m_s,inliers,imu_fused\n")

    print(f"[SYSTEM] {len(left_images)} kare işlenecek. Çıkmak için 'q'.")

    # İlk kare için değişkenler
    prev_rect_l = None
    prev_pts    = None
    prev_3d     = None
    prev_ts_ns  = None

    # İstatistik sayaçları
    n_total   = 0
    n_success = 0
    n_skipped_speed  = 0
    n_skipped_points = 0

    for i in range(len(left_images) - 1):
        n_total += 1

        # ── Timestamp ────────────────────────────────────────────────────
        ts_t0_ns = int(os.path.basename(left_images[i]).split('.')[0])
        ts_t1_ns = int(os.path.basename(left_images[i+1]).split('.')[0])
        dt = (ts_t1_ns - ts_t0_ns) * 1e-9   # saniyeye çevir

        if dt <= 0:
            continue

        # ── Görüntü yükle ────────────────────────────────────────────────
        img_left_t0  = cv2.imread(left_images[i],    cv2.IMREAD_GRAYSCALE)
        img_right_t0 = cv2.imread(right_images[i],   cv2.IMREAD_GRAYSCALE)
        img_left_t1  = cv2.imread(left_images[i+1],  cv2.IMREAD_GRAYSCALE)

        if img_left_t0 is None or img_right_t0 is None or img_left_t1 is None:
            continue

        # ── ADIM 1: Stereo → Derinlik + Rectified noktalar ───────────────
        pts_t0, pts_3d_t0, rect_l_t0 = tracker.process_space_get_depth(
            img_left_t0, img_right_t0
        )

        if len(pts_t0) < 6:
            n_skipped_points += 1
            continue

        # ── IMU entegrasyonu (bu kare aralığındaki örnekler) ─────────────
        imu_fused = False
        if use_imu:
            tracker.imu.reset()
            imu_samples = get_imu_between(imu_data, ts_t0_ns, ts_t1_ns)

            if len(imu_samples) >= 2:
                for j in range(len(imu_samples) - 1):
                    ts_a, gyro_a, accel_a = imu_samples[j]
                    ts_b, _,      _       = imu_samples[j+1]
                    dt_imu = (ts_b - ts_a) * 1e-9
                    tracker.imu.integrate(gyro_a, accel_a, dt_imu)
                imu_fused = True

        # ── ADIM 2: Optik akış (rectified uzayda) ────────────────────────
        # rect_l_t1: t1 anının rectified görüntüsünü al
        _, _, rect_l_t1 = tracker.process_space_get_depth(img_left_t1, img_right_t0)
        # Not: rect_l_t1 hesaplamak için sağ görüntü lazım değil,
        # sadece sol kanalın rectify'ını alıyoruz.
        # Daha verimli yöntem: rectify map'i doğrudan uygula
        h, w = img_left_t1.shape[:2]
        tracker._ensure_maps(h, w)
        rect_l_t1_direct = cv2.remap(
            img_left_t1, tracker._ml1, tracker._ml2, cv2.INTER_LINEAR
        )

        p0_tracked, p1_tracked, _ = tracker.track_time_get_flow(
            rect_l_t0, rect_l_t1_direct, pts_t0
        )

        if len(p0_tracked) < 6:
            n_skipped_points += 1
            continue

        # ── ADIM 3: PnP odometry ─────────────────────────────────────────
        # p0_tracked noktalarına karşılık gelen 3D noktaları bul
        # (pts_t0 ile p0_tracked aynı sırada değil; eşleştir)
        p0_arr = np.array(p0_tracked)
        pt_arr = pts_t0.reshape(-1, 2)

        matched_3d = []
        matched_2d = []
        for k, p0 in enumerate(p0_arr):
            dists = np.linalg.norm(pt_arr - p0, axis=1)
            idx   = np.argmin(dists)
            if dists[idx] < 1.5:   # 1.5 piksel tolerans
                matched_3d.append(pts_3d_t0[idx])
                matched_2d.append(p1_tracked[k])

        if len(matched_3d) < 6:
            n_skipped_points += 1
            continue

        rvec, tvec, inliers = tracker.calculate_odometry(
            np.array(matched_3d), np.array(matched_2d)
        )

        if tvec is None:
            continue

        # ── ADIM 4: IMU füzyonu + hız hesabı ─────────────────────────────
        _, _, imu_delta_p = tracker.imu.get_prediction()

        if imu_fused:
            tvec = tracker.fuse_imu_vision(tvec, imu_delta_p, len(inliers))

        speed = np.linalg.norm(tvec) / dt

        # ── DÜZELTME 5: Fiziksel kısıt (drone için 4 m/s yeterli) ────────
        if speed > 4.0:
            n_skipped_speed += 1
            continue

        n_success += 1
        log_file.write(
            f"{ts_t1_ns},"
            f"{tvec[0][0]:.6f},{tvec[1][0]:.6f},{tvec[2][0]:.6f},"
            f"{speed:.4f},{len(inliers)},{int(imu_fused)}\n"
        )

        # ── Görselleştirme ───────────────────────────────────────────────
        display = cv2.cvtColor(rect_l_t1_direct, cv2.COLOR_GRAY2BGR)

        for pt in p1_tracked:
            cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        # İnlier noktaları kırmızı ile işaretle
        if inliers is not None:
            for idx in inliers.ravel():
                if idx < len(matched_2d):
                    pt = matched_2d[idx]
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), 1)

        cv2.putText(display,
            f"Frame: {i+1}/{len(left_images)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(display,
            f"Speed: {speed:.3f} m/s",
            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
        cv2.putText(display,
            f"Inliers: {len(inliers)}  |  Pts: {len(matched_3d)}",
            (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 0), 2)
        cv2.putText(display,
            f"IMU: {'ON' if imu_fused else 'OFF'}",
            (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 255, 0) if imu_fused else (100, 100, 100), 2)
        cv2.putText(display,
            f"tx={tvec[0][0]:.3f} ty={tvec[1][0]:.3f} tz={tvec[2][0]:.3f}",
            (10, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("VIO — Stereo + IMU", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Özet ─────────────────────────────────────────────────────────────
    log_file.close()
    cv2.destroyAllWindows()

    print("\n" + "="*55)
    print(f"  Toplam kare        : {n_total}")
    print(f"  Başarılı tahmin    : {n_success}")
    print(f"  Hız filtresi çıktı : {n_skipped_speed}")
    print(f"  Nokta yetersiz     : {n_skipped_points}")
    print(f"  Başarı oranı       : {100*n_success/max(n_total,1):.1f}%")
    print("="*55)
    print("  'estimated_trajectory.csv' oluşturuldu.")
    print("="*55)


if __name__ == "__main__":
    main()