import numpy as np
import cv2 as cv
import pandas as pd
import os

# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================
CONFIG = {
    'CAM0_CSV': 'cam0/data.csv',
    'CAM0_DIR': 'cam0/data',
    'CAM1_CSV': 'cam1/data.csv',
    'CAM1_DIR': 'cam1/data',
    'IMU_CSV_PATH': 'imu0/data.csv',
    'GT_CSV_PATH': 'state_groundtruth_estimate0/data.csv',
    
    'FOCAL_LENGTH_PX': 458.654,
    'CX': 376.0, 
    'CY': 240.0, 
    'BASELINE_M': 0.11,
    
    'LK_PARAMS': dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
}

K = np.array([[CONFIG['FOCAL_LENGTH_PX'], 0, CONFIG['CX']],
              [0, CONFIG['FOCAL_LENGTH_PX'], CONFIG['CY']],
              [0, 0, 1]], dtype=np.float64)

# =============================================================================
# 2. VIO DATA LOADER
# =============================================================================
class VIODataLoader:
    def __init__(self):
        df0 = pd.read_csv(CONFIG['CAM0_CSV'], names=['timestamp', 'filename'], header=0)
        df1 = pd.read_csv(CONFIG['CAM1_CSV'], names=['timestamp', 'filename'], header=0)
        self.imu_df = pd.read_csv(CONFIG['IMU_CSV_PATH'])
        self.imu_df.columns = self.imu_df.columns.str.strip()
        self.gt_df = pd.read_csv(CONFIG['GT_CSV_PATH'])
        self.gt_df.columns = self.gt_df.columns.str.strip()
        
        df0 = df0.sort_values('timestamp')
        df1 = df1.sort_values('timestamp')
        self.stereo_df = pd.merge_asof(df0, df1, on='timestamp', direction='nearest', suffixes=('_left', '_right'))
        self.current_idx = 0
        self.total_frames = len(self.stereo_df)

    def get_frame_data(self):
        if self.current_idx >= self.total_frames - 1:
            return None
        row = self.stereo_df.iloc[self.current_idx]
        t_cam = row['timestamp']
        t_next = self.stereo_df.iloc[self.current_idx + 1]['timestamp']
        dt_seconds = (t_next - t_cam) / 1e9
        
        img_left = cv.imread(os.path.join(CONFIG['CAM0_DIR'], str(row['filename_left'])), cv.IMREAD_GRAYSCALE)
        img_right = cv.imread(os.path.join(CONFIG['CAM1_DIR'], str(row['filename_right'])), cv.IMREAD_GRAYSCALE)
        
        imu_idx = (np.abs(self.imu_df['#timestamp [ns]'] - t_cam)).argmin()
        imu_row = self.imu_df.iloc[imu_idx]
        gyro_cam_x = -imu_row['w_RS_S_y [rad s^-1]'] 
        gyro_cam_y = -imu_row['w_RS_S_z [rad s^-1]']
        gyro_cam_z = imu_row['w_RS_S_x [rad s^-1]']
        gyro_vector = np.array([gyro_cam_x, gyro_cam_y, gyro_cam_z])
        
        gt_idx = (np.abs(self.gt_df['#timestamp'] - t_cam)).argmin()
        gt_row = self.gt_df.iloc[gt_idx]
        true_speed = np.sqrt(gt_row['v_RS_R_x [m s^-1]']**2 + gt_row['v_RS_R_y [m s^-1]']**2 + gt_row['v_RS_R_z [m s^-1]']**2)

        self.current_idx += 1
        return img_left, img_right, dt_seconds, gyro_vector, true_speed

# =============================================================================
# 3. WLS STEREO MATCHER
# =============================================================================
def init_wls_stereo_matcher():
    left_matcher = cv.StereoSGBM_create(minDisparity=0, numDisparities=80, blockSize=5, P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=100, speckleRange=32, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    return left_matcher, right_matcher, wls_filter

# =============================================================================
# 4. ANA DÖNGÜ (Keyframe Stratejisi)
# =============================================================================
def main():
    loader = VIODataLoader()
    left_matcher, right_matcher, wls_filter = init_wls_stereo_matcher()
    
    # Durum Değişkenleri (State Variables)
    need_new_keyframe = True
    kf_obj_pts_3d = []
    kf_img_pts_2d = []
    kf_time_elapsed = 0.0
    kf_gyro_accum = np.zeros(3, dtype=np.float64)
    old_left = None

    while True:
        data = loader.get_frame_data()
        if data is None: break
        curr_left, curr_right, dt, gyro_vector, true_speed = data
        vis_frame = cv.cvtColor(curr_left, cv.COLOR_GRAY2BGR)

        # -----------------------------------------------------------------
        # ANA KARE (KEYFRAME) OLUŞTURMA
        # -----------------------------------------------------------------
        if need_new_keyframe:
            # 1. Yeni noktalar bul
            old_p0 = cv.goodFeaturesToTrack(curr_left, mask=None, maxCorners=150, qualityLevel=0.1, minDistance=7)
            
            # 2. Sadece bu karede derinlik hesapla
            left_disp = left_matcher.compute(curr_left, curr_right)
            right_disp = right_matcher.compute(curr_right, curr_left)
            disp_map = wls_filter.filter(left_disp, curr_left, None, right_disp).astype(np.float32) / 16.0
            
            kf_obj_pts_3d = []
            kf_img_pts_2d = []
            valid_p0 = []
            
            # 3. Noktaları 3B Uzaya Çivile
            if old_p0 is not None:
                for pt in old_p0:
                    u, v = int(pt[0][0]), int(pt[0][1])
                    if 0 <= v < disp_map.shape[0] and 0 <= u < disp_map.shape[1]:
                        d = disp_map[v, u]
                        if d > 1.0:
                            Z = (CONFIG['FOCAL_LENGTH_PX'] * CONFIG['BASELINE_M']) / d
                            X = (u - CONFIG['CX']) * Z / CONFIG['FOCAL_LENGTH_PX']
                            Y = (v - CONFIG['CY']) * Z / CONFIG['FOCAL_LENGTH_PX']
                            if Z < 15.0:
                                kf_obj_pts_3d.append([X, Y, Z])
                                kf_img_pts_2d.append([u, v])
                                valid_p0.append(pt)

            kf_obj_pts_3d = np.array(kf_obj_pts_3d, dtype=np.float64)
            old_p0 = np.array(valid_p0, dtype=np.float32)
            
            # Değişkenleri sıfırla
            kf_time_elapsed = 0.0
            kf_gyro_accum = np.zeros(3, dtype=np.float64)
            need_new_keyframe = False
            old_left = curr_left.copy()
            
            cv.putText(vis_frame, "*** YENI ANA KARE (KEYFRAME) ***", (20, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv.imshow('TUSAS - Keyframe VIO', vis_frame)
            cv.waitKey(10)
            continue

        # -----------------------------------------------------------------
        # NORMAL KARELER (Sadece Takip ve Hız Çıkarımı)
        # -----------------------------------------------------------------
        kf_time_elapsed += dt
        kf_gyro_accum += gyro_vector * dt # Jiroskop verisini entegre et
        
        # 1. Noktaları yeni karede takip et
        curr_p1, st, err = cv.calcOpticalFlowPyrLK(old_left, curr_left, old_p0, None, **CONFIG['LK_PARAMS'])
        
        good_new = curr_p1[st == 1]
        good_old_3d = kf_obj_pts_3d[st.flatten() == 1]
        
        for pt in good_new:
            cv.circle(vis_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        # 2. PnP Çözümü (Ana Kare Haritası vs Anlık 2B Pozisyonlar)
        est_speed = 0.0
        if len(good_old_3d) >= 10:
            success, rvec, tvec, inliers = cv.solvePnPRansac(
                good_old_3d, good_new, K, None, 
                useExtrinsicGuess=True, 
                rvec=kf_gyro_accum.reshape(3,1), 
                tvec=np.zeros((3,1)),
                flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=100, reprojectionError=2.0
            )
            
            if success:
                # tvec, Ana Kare'den bu yana toplam metrik yer değiştirmedir
                translation_magnitude = np.linalg.norm(tvec)
                
                # Stabilize Hız = Toplam Yer Değiştirme / Toplam Geçen Zaman
                if kf_time_elapsed > 0:
                    est_speed = translation_magnitude / kf_time_elapsed
                
                # Ekrana Bilgileri Bas
                cv.putText(vis_frame, f"Stabil VIO Hiz : {est_speed:.2f} m/s", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(vis_frame, f"Gercek Hiz (GT): {true_speed:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                error = abs(est_speed - true_speed)
                cv.putText(vis_frame, f"HATA           : {error:.2f} m/s", (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv.putText(vis_frame, f"Keyframe Suresi: {kf_time_elapsed:.2f} sn", (20, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 3. Ana Kareyi Yenileme Koşulları (Karar Mekanizması)
        # Eğer takip edilen nokta sayısı çok düştüyse veya Ana Kare'den çok uzaklaştıysak
        if len(good_old_3d) < 30 or kf_time_elapsed > 0.4:
            need_new_keyframe = True
        else:
            # Döngüyü bir sonraki adım için hazırla
            old_p0 = good_new.reshape(-1, 1, 2)
            kf_obj_pts_3d = good_old_3d
            old_left = curr_left.copy()

        cv.imshow('TUSAS - Keyframe VIO', vis_frame)
        if cv.waitKey(10) & 0xff == 27: break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
