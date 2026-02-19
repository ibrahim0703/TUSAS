import numpy as np
import cv2 as cv
import pandas as pd
import os

# =============================================================================
# 1. KONFİGÜRASYON (EuRoC MH_01_easy - Zemin Hack)
# =============================================================================
CONFIG = {
    'CAM_CSV_PATH': 'cam0/data.csv',
    'CAM_IMG_DIR': 'cam0/data',
    'IMU_CSV_PATH': 'imu0/data.csv',
    'GT_CSV_PATH': 'state_groundtruth_estimate0/data.csv',

    'FOCAL_LENGTH_PX': 458.654,  # EuRoC cam0 fx

    # --- ZEMİN HACK AYARI ---
    # Görüntünün SADECE ALT %40'ını alacağız (Zemine bakan kısım)
    'ROI_TOP_CROP_PERCENT': 0.60,

    'RANSAC_THRESHOLD': 3.0,
    'LK_PARAMS': dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
}


# =============================================================================
# 2. EUROC DATA LOADER (Senkronizasyon Motoru)
# =============================================================================
class EurocDataLoader:
    def __init__(self):
        print("[SİSTEM] EuRoC Veri Seti Yükleniyor...")
        self.cam_df = pd.read_csv(CONFIG['CAM_CSV_PATH'])
        self.cam_df.columns = ['timestamp', 'filename']

        self.imu_df = pd.read_csv(CONFIG['IMU_CSV_PATH'])
        self.imu_df.columns = self.imu_df.columns.str.strip()

        self.gt_df = pd.read_csv(CONFIG['GT_CSV_PATH'])
        self.gt_df.columns = self.gt_df.columns.str.strip()

        self.total_frames = len(self.cam_df)
        self.current_idx = 0

    def get_next_frame_data(self):
        if self.current_idx >= self.total_frames - 1:
            return None

        cam_row = self.cam_df.iloc[self.current_idx]
        t_cam = cam_row['timestamp']
        img_name = cam_row['filename']
        img_path = os.path.join(CONFIG['CAM_IMG_DIR'], str(img_name))

        t_cam_next = self.cam_df.iloc[self.current_idx + 1]['timestamp']
        dt_seconds = (t_cam_next - t_cam) / 1e9

        frame = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"[HATA] Fotoğraf bulunamadı: {img_path}")
            self.current_idx += 1
            return self.get_next_frame_data()

        # IMU Senkronizasyonu
        imu_idx = (np.abs(self.imu_df['#timestamp [ns]'] - t_cam)).argmin()
        imu_row = self.imu_df.iloc[imu_idx]
        gyro_x = imu_row['w_RS_S_x [rad s^-1]']
        gyro_y = imu_row['w_RS_S_y [rad s^-1]']

        # Ground Truth Senkronizasyonu
        gt_idx = (np.abs(self.gt_df['#timestamp'] - t_cam)).argmin()
        gt_row = self.gt_df.iloc[gt_idx]
        true_vx = gt_row['v_RS_R_x [m s^-1]']

        # EuRoC'ta x ileri, y sol, z yukarıdır. İleri hızı alıyoruz.
        true_vy = gt_row['v_RS_R_y [m s^-1]']
        true_z_height = abs(gt_row['p_RS_R_z [m]'])

        self.current_idx += 1
        return {
            'frame': frame, 'dt': dt_seconds, 'gyro': (gyro_x, gyro_y),
            'ground_truth_v': (true_vx, true_vy), 'agl_height': true_z_height
        }


# =============================================================================
# 3. ROTATION COMPENSATION & OPTICAL FLOW
# =============================================================================
def generate_grid_features(frame, step=20):
    h, w = frame.shape[:2]
    y, x = np.mgrid[10:h - 10:step, 10:w - 10:step].reshape(2, -1).astype(int)
    return np.array(list(zip(x, y)), dtype=np.float32).reshape(-1, 1, 2)


def main():
    dataset = EurocDataLoader()
    data = dataset.get_next_frame_data()
    if data is None: return

    # ZEMİN ROI KESİMİ
    h_raw, w_raw = data['frame'].shape
    roi_top = int(h_raw * CONFIG['ROI_TOP_CROP_PERCENT'])

    old_gray = data['frame'][roi_top:h_raw, 0:w_raw]
    p0 = generate_grid_features(old_gray, step=30)

    while True:
        data = dataset.get_next_frame_data()
        if data is None: break

        frame_full = data['frame']
        frame_gray = frame_full[roi_top:h_raw, 0:w_raw]  # Sadece zemin
        dt = data['dt']
        gyro_x, gyro_y = data['gyro']
        true_vx, true_vy = data['ground_truth_v']
        agl = data['agl_height']

        vis_frame = cv.cvtColor(frame_full, cv.COLOR_GRAY2BGR)
        cv.rectangle(vis_frame, (0, roi_top), (w_raw, h_raw), (0, 50, 0), 2)  # Analiz alanını çiz

        if p0 is None or len(p0) < 10:
            p0 = generate_grid_features(frame_gray, step=30)
            old_gray = frame_gray.copy()
            continue

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **CONFIG['LK_PARAMS'])

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) < 4:
                p0 = generate_grid_features(frame_gray, step=30)
                old_gray = frame_gray.copy()
                continue

            M, mask = cv.findHomography(good_old, good_new, cv.RANSAC, CONFIG['RANSAC_THRESHOLD'])

            if M is not None:
                matchesMask = mask.ravel().tolist()
                inlier_dx, inlier_dy = [], []

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    if matchesMask[i]:
                        inlier_dx.append(new[0] - old[0])
                        inlier_dy.append(new[1] - old[1])
                        # Görselleştirmede noktaları doğru yere çizmek için roi_top ekliyoruz
                        cv.line(vis_frame, (int(new[0]), int(new[1] + roi_top)), (int(old[0]), int(old[1] + roi_top)),
                                (0, 255, 0), 1)

                if len(inlier_dx) > 0:
                    mean_dx = np.mean(inlier_dx)
                    mean_dy = np.mean(inlier_dy)

                    # 1. HAM HIZ (Piksel/sn)
                    v_px_raw_x = mean_dx / dt
                    v_px_raw_y = mean_dy / dt

                    # 2. GYRO ROTASYON ETKİSİ
                    # DİKKAT: Bu işaretler test edilip değiştirilecek!
                    v_px_rot_x = gyro_x * CONFIG['FOCAL_LENGTH_PX']
                    v_px_rot_y = gyro_y * CONFIG['FOCAL_LENGTH_PX']

                    # 3. DÜZELTİLMİŞ PİKSEL HIZI (Translation)
                    v_px_trans_x = v_px_raw_x - v_px_rot_x
                    v_px_trans_y = v_px_raw_y - v_px_rot_y

                    # 4. METRİK HIZ (Fiziği oturtuyoruz)
                    # Zemin hacki yaptığımız için, piksellerin kayması derinliğe (AGL) bağlıdır.
                    # Eğer drone ileri giderse, zemin pikselleri EKSİ Y (aşağı) kayar. Bu yüzden -1 ile çarpıyoruz.
                    est_vy = -1 * (v_px_trans_y * agl) / CONFIG['FOCAL_LENGTH_PX']
                    est_vx = (v_px_trans_x * agl) / CONFIG['FOCAL_LENGTH_PX']

                    # Ekrana Yazdır
                    cv.putText(vis_frame, f"Tahmin Ileri Hiz: {est_vy:.2f} m/s", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 255, 0), 2)
                    cv.putText(vis_frame, f"Gercek Ileri Hiz (X): {true_vx:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX,
                               0.6, (0, 0, 255), 2)

                    error = abs(est_vy - true_vx)
                    cv.putText(vis_frame, f"HATA: {error:.2f} m/s", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 255, 255), 2)

                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            else:
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = generate_grid_features(frame_gray, step=30)
            old_gray = frame_gray.copy()

        cv.imshow('TUSAS - Floor ROI Sensor Fusion', vis_frame)
        if cv.waitKey(1) & 0xff == 27: break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
