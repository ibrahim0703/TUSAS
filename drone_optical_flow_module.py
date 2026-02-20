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
# 2. VIO DATA LOADER (Kameralar + IMU + Ground Truth)
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
        
        # --- IMU GYRO SENKRONİZASYONU ---
        imu_idx = (np.abs(self.imu_df['#timestamp [ns]'] - t_cam)).argmin()
        imu_row = self.imu_df.iloc[imu_idx]
        
        # EuRoC IMU'su: X İleri, Y Sol, Z Yukarı. 
        # Bunu Kamera Eksenine (Z İleri, X Sağ, Y Aşağı) çevirmemiz lazım.
        gyro_x_imu = imu_row['w_RS_S_x [rad s^-1]'] # Roll
        gyro_y_imu = imu_row['w_RS_S_y [rad s^-1]'] # Pitch
        gyro_z_imu = imu_row['w_RS_S_z [rad s^-1]'] # Yaw
        
        # Basit Eksen Dönüşümü (Kamera Eksenleri için)
        gyro_cam_x = -gyro_y_imu 
        gyro_cam_y = -gyro_z_imu
        gyro_cam_z = gyro_x_imu
        
        gyro_vector = np.array([gyro_cam_x, gyro_cam_y, gyro_cam_z])
        
        # Ground Truth
        gt_idx = (np.abs(self.gt_df['#timestamp'] - t_cam)).argmin()
        gt_row = self.gt_df.iloc[gt_idx]
        true_speed = np.sqrt(gt_row['v_RS_R_x [m s^-1]']**2 + gt_row['v_RS_R_y [m s^-1]']**2 + gt_row['v_RS_R_z [m s^-1]']**2)

        self.current_idx += 1
        return img_left, img_right, dt_seconds, gyro_vector, true_speed

# =============================================================================
# 3. WLS STEREO MATCHER
# =============================================================================
def init_wls_stereo_matcher():
    left_matcher = cv.StereoSGBM_create(
        minDisparity=0, numDisparities=80, blockSize=5,
        P1=8 * 1 * 25, P2=32 * 1 * 25,
        disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=100, speckleRange=32,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    return left_matcher, right_matcher, wls_filter

# =============================================================================
# 4. ANA DÖNGÜ (IMU Güçlendirmeli PnP)
# =============================================================================
def main():
    loader = VIODataLoader()
    left_matcher, right_matcher, wls_filter = init_wls_stereo_matcher()
    
    data = loader.get_frame_data()
    if data is None: return
    old_left, old_right, _, _, _ = data
    
    old_p0 = cv.goodFeaturesToTrack(old_left, mask=None, maxCorners=150, qualityLevel=0.1, minDistance=7)
    
    left_disp = left_matcher.compute(old_left, old_right)
    right_disp = right_matcher.compute(old_right, old_left)
    old_disp = wls_filter.filter(left_disp, old_left, None, right_disp).astype(np.float32) / 16.0

    while True:
        data = loader.get_frame_data()
        if data is None: break
        curr_left, curr_right, dt, gyro_vector, true_speed = data
        
        vis_frame = cv.cvtColor(curr_left, cv.COLOR_GRAY2BGR)

        if old_p0 is not None and len(old_p0) > 10:
            curr_p1, st, err = cv.calcOpticalFlowPyrLK(old_left, curr_left, old_p0, None, **CONFIG['LK_PARAMS'])
            
            good_new = curr_p1[st == 1]
            good_old = old_p0[st == 1]
            
            obj_pts_3d = []
            img_pts_2d = []
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                u_old, v_old = int(old[0]), int(old[1])
                if 0 <= v_old < old_disp.shape[0] and 0 <= u_old < old_disp.shape[1]:
                    d = old_disp[v_old, u_old]
                    if d > 1.0:
                        Z = (CONFIG['FOCAL_LENGTH_PX'] * CONFIG['BASELINE_M']) / d
                        X = (u_old - CONFIG['CX']) * Z / CONFIG['FOCAL_LENGTH_PX']
                        Y = (v_old - CONFIG['CY']) * Z / CONFIG['FOCAL_LENGTH_PX']
                        
                        if Z < 15.0:
                            obj_pts_3d.append([X, Y, Z])
                            img_pts_2d.append([new[0], new[1]])
                            cv.circle(vis_frame, (int(new[0]), int(new[1])), 3, (0, 255, 0), -1)

            obj_pts_3d = np.array(obj_pts_3d, dtype=np.float64)
            img_pts_2d = np.array(img_pts_2d, dtype=np.float64)

            if len(obj_pts_3d) >= 6:
                
                # --- İŞTE SİHİR BURADA BAŞLIYOR (IMU KOPYASI) ---
                # Gyro (Radyan/sn) verisini, dt ile çarparak bu iki kare arasındaki toplam Dönüş Açısına çeviriyoruz.
                guess_rvec = gyro_vector * dt
                # Translation için bir fikrimiz yok, sıfır veriyoruz.
                guess_tvec = np.zeros((3, 1), dtype=np.float64)
                
                # PnP'ye "Ben sana rvec verdim, sen onu kullan, kafana göre dönme arama" diyoruz.
                success, rvec, tvec, inliers = cv.solvePnPRansac(
                    obj_pts_3d, img_pts_2d, K, None, 
                    useExtrinsicGuess=True, # TUSAŞ VIO standardı: Ön tahmin kullan!
                    rvec=guess_rvec.reshape(3,1), 
                    tvec=guess_tvec,
                    flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=100, reprojectionError=2.0
                )
                
                if success:
                    translation_magnitude = np.linalg.norm(tvec)
                    est_speed = translation_magnitude / dt
                    
                    cv.putText(vis_frame, f"VIO Hiz (IMU+PnP): {est_speed:.2f} m/s", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv.putText(vis_frame, f"Gercek Hiz (GT)  : {true_speed:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    error = abs(est_speed - true_speed)
                    cv.putText(vis_frame, f"HATA: {error:.2f} m/s", (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            old_left = curr_left.copy()
            left_disp = left_matcher.compute(curr_left, curr_right)
            right_disp = right_matcher.compute(curr_right, curr_left)
            old_disp = wls_filter.filter(left_disp, curr_left, None, right_disp).astype(np.float32) / 16.0
            old_p0 = cv.goodFeaturesToTrack(curr_left, mask=None, maxCorners=150, qualityLevel=0.1, minDistance=7)

        cv.imshow('TUSAS - Tightly Coupled VIO (IMU Gyro + PnP)', vis_frame)
        if cv.waitKey(10) & 0xff == 27: break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
