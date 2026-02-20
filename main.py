import numpy as np
import cv2 as cv
import pandas as pd
import os
from config import CONFIG, K, R_CB
from eskf import ESKF

# =============================================================================
# 1. VIO DATA LOADER
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
        
        w_B = np.array([imu_row['w_RS_S_x [rad s^-1]'], imu_row['w_RS_S_y [rad s^-1]'], imu_row['w_RS_S_z [rad s^-1]']])
        gyro_cam = R_CB @ w_B
        
        a_B = np.array([imu_row['a_RS_S_x [m s^-2]'], imu_row['a_RS_S_y [m s^-2]'], imu_row['a_RS_S_z [m s^-2]']])
        accel_cam = R_CB @ a_B
        
        gt_idx = (np.abs(self.gt_df['#timestamp'] - t_cam)).argmin()
        gt_row = self.gt_df.iloc[gt_idx]
        true_speed = np.sqrt(gt_row['v_RS_R_x [m s^-1]']**2 + gt_row['v_RS_R_y [m s^-1]']**2 + gt_row['v_RS_R_z [m s^-1]']**2)

        self.current_idx += 1
        return img_left, img_right, dt_seconds, accel_cam, gyro_cam, true_speed

def init_wls_stereo_matcher():
    left_matcher = cv.StereoSGBM_create(minDisparity=0, numDisparities=80, blockSize=5, P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=100, speckleRange=32, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    return left_matcher, right_matcher, wls_filter

def main():
    loader = VIODataLoader()
    left_matcher, right_matcher, wls_filter = init_wls_stereo_matcher()
    filter_eskf = ESKF() 
    
    # -----------------------------------------------------------------
    # EKSİK OLAN GERÇEK FİZİK: SENSÖR ISITMA VE HİZALAMA FAZI
    # -----------------------------------------------------------------
    print("[SİSTEM] Sensörler ısınıyor, Yerçekimi hizalanıyor. Bekleyin...")
    accel_buffer, gyro_buffer = [], []
    for _ in range(15): # İlk 15 kareyi (~0.7 saniye) veriyi toplamak için kullan
        data = loader.get_frame_data()
        if data is None: return
        accel_buffer.append(data[3])
        gyro_buffer.append(data[4])
    
    filter_eskf.initialize_system(np.array(accel_buffer), np.array(gyro_buffer))
    print("[SİSTEM] Hizalama Tamamlandı. VIO Döngüsü Başlıyor!")
    
    need_new_keyframe = True
    kf_obj_pts_3d = []
    kf_gyro_accum = np.zeros(3, dtype=np.float64)
    kf_time_elapsed = 0.0 
    old_left = None

    while True:
        data = loader.get_frame_data()
        if data is None: break
        curr_left, curr_right, dt, accel_cam, gyro_cam, true_speed = data
        vis_frame = cv.cvtColor(curr_left, cv.COLOR_GRAY2BGR)

        filter_eskf.predict(accel_cam, gyro_cam, dt)

        if need_new_keyframe:
            old_p0 = cv.goodFeaturesToTrack(curr_left, mask=None, maxCorners=150, qualityLevel=0.1, minDistance=7)
            left_disp = left_matcher.compute(curr_left, curr_right)
            right_disp = right_matcher.compute(curr_right, curr_left)
            disp_map = wls_filter.filter(left_disp, curr_left, None, right_disp).astype(np.float32) / 16.0
            
            kf_obj_pts_3d = []
            valid_p0 = []
            
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
                                valid_p0.append(pt)

            kf_obj_pts_3d = np.array(kf_obj_pts_3d, dtype=np.float64)
            old_p0 = np.array(valid_p0, dtype=np.float32)
            
            kf_gyro_accum = np.zeros(3, dtype=np.float64)
            kf_time_elapsed = 0.0 
            need_new_keyframe = False
            old_left = curr_left.copy()
            continue

        kf_time_elapsed += dt
        kf_gyro_accum += gyro_cam * dt
        
        curr_p1, st, err = cv.calcOpticalFlowPyrLK(old_left, curr_left, old_p0, None, **CONFIG['LK_PARAMS'])
        good_new = curr_p1[st == 1]
        good_old_3d = kf_obj_pts_3d[st.flatten() == 1]
        
        for pt in good_new:
            cv.circle(vis_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        if len(good_old_3d) >= 10:
            success, rvec, tvec, inliers = cv.solvePnPRansac(
                good_old_3d, good_new, K, None, 
                useExtrinsicGuess=True, 
                rvec=kf_gyro_accum.reshape(3,1), 
                tvec=np.zeros((3,1)),
                flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=100, reprojectionError=2.0
            )
            
            if success and kf_time_elapsed > 0:
                # -----------------------------------------------------------------
                # İŞTE KÖTÜ VEKTÖR MATEMATİĞİNİN DÜZELTİLDİĞİ YER
                # -----------------------------------------------------------------
                # 1. Rotasyon Vektörünü (rvec) 3x3 Matrise çevir
                R_pnp, _ = cv.Rodrigues(rvec)
                
                # 2. PnP'nin ters dünyasını düzelt: P_cam = -R^T * tvec
                # Bu bize kameranın Keyframe uzayındaki GERÇEK konumunu verir.
                t_cam_kf = -R_pnp.T @ tvec
                
                # 3. Kameranın Gövde Hızını Bul
                v_cam_measured = t_cam_kf.flatten() / kf_time_elapsed
                
                # 4. Kamera Ekseninden -> IMU Gövde Eksenine geç
                v_imu_body = R_CB.T @ v_cam_measured
                
                # 5. Gövde Ekseninden -> Dünya Eksenine geç
                v_world_measured = filter_eskf.R @ v_imu_body
                
                # Ve HIZI FİLTREYE BESLE!
                filter_eskf.update_velocity(v_world_measured)
                
                kalman_speed = filter_eskf.get_speed()
                cv.putText(vis_frame, f"ESKF Hiz: {kalman_speed:.2f} m/s", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(vis_frame, f"GT Hiz  : {true_speed:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                error = abs(kalman_speed - true_speed)
                cv.putText(vis_frame, f"HATA    : {error:.2f} m/s", (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if len(good_old_3d) < 30 or kf_time_elapsed > 0.4:
            need_new_keyframe = True
        else:
            old_p0 = good_new.reshape(-1, 1, 2)
            kf_obj_pts_3d = good_old_3d
            old_left = curr_left.copy()

        cv.imshow('TUSAS - Master VIO', vis_frame)
        if cv.waitKey(10) & 0xff == 27: break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
