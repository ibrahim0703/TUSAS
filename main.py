import numpy as np
import cv2 as cv
import pandas as pd
import os
from config import CONFIG, K, R_CB
from eskf import ESKF

# =============================================================================
# 1. EUROC KALİBRASYON (STEREO RECTIFICATION)
# =============================================================================
K0 = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
D0 = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
K1 = np.array([[457.587, 0.0, 379.999], [0.0, 456.134, 255.238], [0.0, 0.0, 1.0]])
D1 = np.array([-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05])
R_stereo = np.array([[0.999997, 0.002521, -0.000670], [-0.002519, 0.999993, 0.002758], [0.000677, -0.002757, 0.999996]])
T_stereo = np.array([-0.110074, 0.000399, -0.000853])

image_size = (752, 480)
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(K0, D0, K1, D1, image_size, R_stereo, T_stereo, alpha=0)
map1_x, map1_y = cv.initUndistortRectifyMap(K0, D0, R1, P1, image_size, cv.CV_32FC1)
map2_x, map2_y = cv.initUndistortRectifyMap(K1, D1, R2, P2, image_size, cv.CV_32FC1)

# =============================================================================
# 2. FULL VIO DATA LOADER (200 HZ IMU BATCH PROCESSING)
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
        if self.current_idx >= self.total_frames: return None
        
        row_curr = self.stereo_df.iloc[self.current_idx]
        t_cam_curr = row_curr['timestamp']
        
        # Bir önceki kameranın zamanını bul (IMU aralığını hesaplamak için)
        if self.current_idx == 0:
            t_cam_prev = t_cam_curr - int(0.05 * 1e9) 
        else:
            t_cam_prev = self.stereo_df.iloc[self.current_idx - 1]['timestamp']
            
        raw_left = cv.imread(os.path.join(CONFIG['CAM0_DIR'], str(row_curr['filename_left'])), cv.IMREAD_GRAYSCALE)
        raw_right = cv.imread(os.path.join(CONFIG['CAM1_DIR'], str(row_curr['filename_right'])), cv.IMREAD_GRAYSCALE)
        rect_left = cv.remap(raw_left, map1_x, map1_y, cv.INTER_LINEAR)
        rect_right = cv.remap(raw_right, map2_x, map2_y, cv.INTER_LINEAR)
        
        # Önceki kare ile şu anki kare arasında kalan TÜM 200Hz IMU verilerini çek
        imu_mask = (self.imu_df['#timestamp [ns]'] > t_cam_prev) & (self.imu_df['#timestamp [ns]'] <= t_cam_curr)
        imu_batch = self.imu_df[imu_mask]
        
        gt_idx = (np.abs(self.gt_df['#timestamp'] - t_cam_curr)).argmin()
        gt_row = self.gt_df.iloc[gt_idx]
        true_speed = np.sqrt(gt_row['v_RS_R_x [m s^-1]']**2 + gt_row['v_RS_R_y [m s^-1]']**2 + gt_row['v_RS_R_z [m s^-1]']**2)

        self.current_idx += 1
        return rect_left, rect_right, imu_batch, true_speed

# =============================================================================
# 3. GÖRÜNTÜ İŞLEME FONKSİYONLARI
# =============================================================================
def init_wls_stereo_matcher():
    left_matcher = cv.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=11, 
        P1=8*1*11**2, P2=32*1*11**2, disp12MaxDiff=10, 
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=32, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    return left_matcher, right_matcher, wls_filter

def get_distributed_features(img, max_corners=300, grid_size=(4, 4)):
    img_blur = cv.GaussianBlur(img, (3, 3), 0)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_blur)
    h, w = img_clahe.shape
    grid_h, grid_w = h // grid_size[0], w // grid_size[1]
    pts_per_grid = max_corners // (grid_size[0] * grid_size[1])
    distributed_pts = []

    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            y0, y1 = r * grid_h, (r + 1) * grid_h
            x0, x1 = c * grid_w, (c + 1) * grid_w
            grid_roi = img_clahe[y0:y1, x0:x1]
            corners = cv.goodFeaturesToTrack(grid_roi, maxCorners=pts_per_grid, qualityLevel=0.02, minDistance=5)
            if corners is not None:
                for pt in corners:
                    pt[0][0] += x0
                    pt[0][1] += y0
                    distributed_pts.append(pt)

    if len(distributed_pts) == 0: return None
    return np.array(distributed_pts, dtype=np.float32).reshape(-1, 1, 2)

# =============================================================================
# 4. MASTER VIO DÖNGÜSÜ (ESKF + VISION)
# =============================================================================
def main():
    loader = VIODataLoader()
    left_matcher, right_matcher, wls_filter = init_wls_stereo_matcher()
    filter_eskf = ESKF() 
    
    print("[SİSTEM] Sensorler isiniyor, lutfen bekleyin...")
    accel_buffer, gyro_buffer = [], []
    # Statik kalibrasyon için ilk 2 kameranın IMU verilerini topla (Yaklaşık 20 veri)
    for _ in range(2): 
        data = loader.get_frame_data()
        if data is None: return
        for _, row in data[2].iterrows():
            w_B = np.array([row['w_RS_S_x [rad s^-1]'], row['w_RS_S_y [rad s^-1]'], row['w_RS_S_z [rad s^-1]']])
            a_B = np.array([row['a_RS_S_x [m s^-2]'], row['a_RS_S_y [m s^-2]'], row['a_RS_S_z [m s^-2]']])
            gyro_buffer.append(R_CB @ w_B)
            accel_buffer.append(R_CB @ a_B)
            
    filter_eskf.initialize_system(np.array(accel_buffer), np.array(gyro_buffer))
    print("[SİSTEM] VIO Basladi. 200 Hz IMU Entegrasyonu ve Eksen Donusumleri Devrede.")
    
    need_new_keyframe = True
    kf_obj_pts_3d = []
    kf_gyro_accum = np.zeros(3, dtype=np.float64)
    kf_time_elapsed = 0.0 
    kf_R_WB = np.eye(3) 
    old_left = None
    old_p0 = None

    while True:
        data = loader.get_frame_data()
        if data is None: break
        curr_left, curr_right, imu_batch, true_speed = data
        vis_frame = cv.cvtColor(curr_left, cv.COLOR_GRAY2BGR)

        # ---------------------------------------------------------------------
        # 1. ESKF SÜREKLİ TAHMİN (PREDICT) - 200 Hz IMU BATCH İŞLEME
        # ---------------------------------------------------------------------
        last_imu_time = imu_batch.iloc[0]['#timestamp [ns]'] if len(imu_batch) > 0 else 0
        
        for index, imu_row in imu_batch.iterrows():
            current_imu_time = imu_row['#timestamp [ns]']
            dt_imu = (current_imu_time - last_imu_time) / 1e9
            if dt_imu <= 0: dt_imu = 0.005 # Güvenlik önlemi (0 dt ile patlamaması için)
            
            w_B = np.array([imu_row['w_RS_S_x [rad s^-1]'], imu_row['w_RS_S_y [rad s^-1]'], imu_row['w_RS_S_z [rad s^-1]']])
            gyro_cam = R_CB @ w_B
            
            a_B = np.array([imu_row['a_RS_S_x [m s^-2]'], imu_row['a_RS_S_y [m s^-2]'], imu_row['a_RS_S_z [m s^-2]']])
            accel_cam = R_CB @ a_B
            
            # Kalman Filtresini ufak 0.005 sn'lik adımlarla ilerlet (Gerçek Fizik)
            filter_eskf.predict(accel_cam, gyro_cam, dt_imu)
            
            # PnP algoritmasının ihtiyacı olan rotasyon ve zaman birikimini güncelle
            kf_gyro_accum += gyro_cam * dt_imu
            kf_time_elapsed += dt_imu
            last_imu_time = current_imu_time

        # ---------------------------------------------------------------------
        # 2. ANA KARE OLUŞTURMA VE DERİNLİK (KEYFRAME) - 20 Hz
        # ---------------------------------------------------------------------
        if need_new_keyframe:
            old_p0 = get_distributed_features(curr_left, max_corners=300)
            
            if old_p0 is not None:
                left_disp = left_matcher.compute(curr_left, curr_right)
                right_disp = right_matcher.compute(curr_right, curr_left)
                disp_map = wls_filter.filter(left_disp, curr_left, None, right_disp).astype(np.float32) / 16.0
                
                disp_vis = np.zeros_like(disp_map, dtype=np.uint8)
                mask = (disp_map > 0).astype(np.uint8) 
                disp_clipped = np.clip(disp_map, 0, 64) 
                cv.normalize(disp_clipped, disp_vis, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, mask=mask)
                color_disp = cv.applyColorMap(disp_vis, cv.COLORMAP_JET)
                color_disp[mask == 0] = [0, 0, 0] 
                cv.imshow('TUSAS - Derinlik Haritasi', color_disp)
                
                kf_obj_pts_3d = []
                valid_p0 = []
                
                for pt in old_p0:
                    u, v = int(pt[0][0]), int(pt[0][1])
                    if 0 <= v < disp_map.shape[0] and 0 <= u < disp_map.shape[1]:
                        d = disp_map[v, u]
                        if d > 1.0: 
                            Z = (458.654 * 0.110074) / d
                            X = (u - 367.215) * Z / 458.654
                            Y = (v - 248.375) * Z / 458.654
                            
                            if 0.5 < Z < 30.0: 
                                kf_obj_pts_3d.append([X, Y, Z])
                                valid_p0.append(pt)

                kf_obj_pts_3d = np.array(kf_obj_pts_3d, dtype=np.float64)
                old_p0 = np.array(valid_p0, dtype=np.float32).reshape(-1, 1, 2)
                
                kf_time_elapsed = 0.0 
                kf_gyro_accum = np.zeros(3, dtype=np.float64)
                kf_R_WB = filter_eskf.R.copy() # Dünya rotasyonunu mühürle
                
                need_new_keyframe = False
                old_left = curr_left.copy()
            continue

        # ---------------------------------------------------------------------
        # 3. OPTİK AKIŞ VE EPIPOLAR KORUMA
        # ---------------------------------------------------------------------
        curr_p1, st, err = cv.calcOpticalFlowPyrLK(old_left, curr_left, old_p0, None, **CONFIG['LK_PARAMS'])
        
        good_new_raw = curr_p1[st == 1]
        good_old_2d_raw = old_p0[st.flatten() == 1] 
        good_old_3d_raw = kf_obj_pts_3d[st.flatten() == 1]

        if len(good_new_raw) >= 15:
            F_mat, inlier_mask = cv.findFundamentalMat(good_old_2d_raw, good_new_raw, cv.FM_RANSAC, 1.0, 0.99)
            if inlier_mask is not None:
                valid_mask = inlier_mask.flatten() == 1
                good_new = good_new_raw[valid_mask]
                good_old_3d = good_old_3d_raw[valid_mask]
            else:
                good_new, good_old_3d = good_new_raw, good_old_3d_raw
        else:
            good_new, good_old_3d = good_new_raw, good_old_3d_raw

        for pt in good_new:
            cv.circle(vis_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        # ---------------------------------------------------------------------
        # 4. PNP ÇÖZÜMÜ VE ESKF GÜNCELLEME (UPDATE)
        # ---------------------------------------------------------------------
        if len(good_old_3d) >= 15: 
            success, rvec, tvec, inliers = cv.solvePnPRansac(
                good_old_3d, good_new, K, None, 
                useExtrinsicGuess=True,              
                rvec=kf_gyro_accum.reshape(3,1),     
                tvec=np.zeros((3,1)),                
                flags=cv.SOLVEPNP_ITERATIVE, reprojectionError=2.0
            )
            
            if success and kf_time_elapsed > 0:
                R_pnp, _ = cv.Rodrigues(rvec)
                t_cam_kf = -R_pnp.T @ tvec
                
                v_cam_measured = (t_cam_kf.flatten() / kf_time_elapsed) 
                
                # Eksen Dönüşümü: Kamera -> Gövde -> Dünya
                v_body_measured = R_CB.T @ v_cam_measured
                v_world_measured = kf_R_WB @ v_body_measured
                
                # Filtreye hizalanmış gerçek dünya hızını (Ölçümü) veriyoruz
                filter_eskf.update_velocity(v_world_measured)
                
                kalman_speed = filter_eskf.get_speed()
                error = abs(kalman_speed - true_speed)
                
                cv.putText(vis_frame, f"ESKF Hiz: {kalman_speed:.2f} m/s", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(vis_frame, f"GT Hiz  : {true_speed:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
