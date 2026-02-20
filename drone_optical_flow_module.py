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
    'GT_CSV_PATH': 'state_groundtruth_estimate0/data.csv',
    
    'FOCAL_LENGTH_PX': 458.654,
    'CX': 376.0, # EuRoC 752x480 çözünürlüğün X merkezi
    'CY': 240.0, # EuRoC 752x480 çözünürlüğün Y merkezi
    'BASELINE_M': 0.11,
    
    'LK_PARAMS': dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
}

# Kamera İç Parametre Matrisi (Intrinsic Matrix - K)
K = np.array([[CONFIG['FOCAL_LENGTH_PX'], 0, CONFIG['CX']],
              [0, CONFIG['FOCAL_LENGTH_PX'], CONFIG['CY']],
              [0, 0, 1]], dtype=np.float64)

# =============================================================================
# 2. DATA LOADER (Kameralar + Ground Truth)
# =============================================================================
class StereoDataLoader:
    def __init__(self):
        df0 = pd.read_csv(CONFIG['CAM0_CSV'], names=['timestamp', 'filename'], header=0)
        df1 = pd.read_csv(CONFIG['CAM1_CSV'], names=['timestamp', 'filename'], header=0)
        
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
        
        # dt hesaplama
        t_next = self.stereo_df.iloc[self.current_idx + 1]['timestamp']
        dt_seconds = (t_next - t_cam) / 1e9
        
        img_left = cv.imread(os.path.join(CONFIG['CAM0_DIR'], str(row['filename_left'])), cv.IMREAD_GRAYSCALE)
        img_right = cv.imread(os.path.join(CONFIG['CAM1_DIR'], str(row['filename_right'])), cv.IMREAD_GRAYSCALE)
        
        # Ground Truth Senkronizasyonu
        gt_idx = (np.abs(self.gt_df['#timestamp'] - t_cam)).argmin()
        gt_row = self.gt_df.iloc[gt_idx]
        
        # EuRoC eksenlerinde ileri gidiş +X olarak kodlanmıştır (World Frame'e göre)
        # Hız büyüklüğünü (Magnitude) alıyoruz ki yön karmaşası yaşamayalım
        vx = gt_row['v_RS_R_x [m s^-1]']
        vy = gt_row['v_RS_R_y [m s^-1]']
        vz = gt_row['v_RS_R_z [m s^-1]']
        true_speed = np.sqrt(vx**2 + vy**2 + vz**2)

        self.current_idx += 1
        return img_left, img_right, dt_seconds, true_speed

# =============================================================================
# 3. WLS STEREO MATCHER
# =============================================================================
def init_wls_stereo_matcher():
    window_size = 5
    left_matcher = cv.StereoSGBM_create(
        minDisparity=0, numDisparities=16*5, blockSize=window_size,
        P1=8 * 1 * window_size ** 2, P2=32 * 1 * window_size ** 2,
        disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=100, speckleRange=32,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    return left_matcher, right_matcher, wls_filter

# =============================================================================
# 4. ANA DÖNGÜ (PnP Pose Estimation)
# =============================================================================
def main():
    loader = StereoDataLoader()
    left_matcher, right_matcher, wls_filter = init_wls_stereo_matcher()
    
    # 1. İlk Kareyi Hazırla
    old_left, old_right, _, _ = loader.get_frame_data()
    if old_left is None: return
    
    old_p0 = cv.goodFeaturesToTrack(old_left, mask=None, maxCorners=150, qualityLevel=0.1, minDistance=7)
    
    # İlk karenin derinlik haritasını çıkar
    left_disp = left_matcher.compute(old_left, old_right)
    right_disp = right_matcher.compute(old_right, old_left)
    old_disp = wls_filter.filter(left_disp, old_left, None, right_disp).astype(np.float32) / 16.0

    while True:
        data = loader.get_frame_data()
        if data is None: break
        curr_left, curr_right, dt, true_speed = data
        
        vis_frame = cv.cvtColor(curr_left, cv.COLOR_GRAY2BGR)

        # 2. Optik Akış ile noktaları yeni kareye taşı
        if old_p0 is not None and len(old_p0) > 10:
            curr_p1, st, err = cv.calcOpticalFlowPyrLK(old_left, curr_left, old_p0, None, **CONFIG['LK_PARAMS'])
            
            good_new = curr_p1[st == 1]
            good_old = old_p0[st == 1]
            
            # 3. 3D-2D Eşleştirme Dizilerini Hazırla
            obj_pts_3d = [] # Eski karedeki 3B Gerçek Dünya Koordinatları
            img_pts_2d = [] # Yeni karedeki 2B Ekran Koordinatları
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                u_old, v_old = int(old[0]), int(old[1])
                u_new, v_new = new[0], new[1]
                
                if 0 <= v_old < old_disp.shape[0] and 0 <= u_old < old_disp.shape[1]:
                    d = old_disp[v_old, u_old]
                    
                    if d > 1.0: # Çok uzak ve gürültülü noktaları (d<1) alma
                        # --- 3B KOORDİNAT HESABI (Kamera Eksenine Göre) ---
                        Z = (CONFIG['FOCAL_LENGTH_PX'] * CONFIG['BASELINE_M']) / d
                        X = (u_old - CONFIG['CX']) * Z / CONFIG['FOCAL_LENGTH_PX']
                        Y = (v_old - CONFIG['CY']) * Z / CONFIG['FOCAL_LENGTH_PX']
                        
                        if Z < 15.0: # 15 metreden uzak noktalar PnP'yi bozar
                            obj_pts_3d.append([X, Y, Z])
                            img_pts_2d.append([u_new, v_new])
                            
                            # Görselleştirme
                            cv.circle(vis_frame, (int(u_new), int(v_new)), 3, (0, 255, 0), -1)

            obj_pts_3d = np.array(obj_pts_3d, dtype=np.float64)
            img_pts_2d = np.array(img_pts_2d, dtype=np.float64)

            # 4. PnP ile Kameranın Hareketini (Pose) Çöz
            if len(obj_pts_3d) >= 6: # PnP için minimum 6 nokta gerekir (RANSAC ile)
                # Kamera nereye gitti?
                success, rvec, tvec, inliers = cv.solvePnPRansac(
                    obj_pts_3d, img_pts_2d, K, None, 
                    flags=cv.SOLVEPNP_ITERATIVE, iterationsCount=100, reprojectionError=2.0
                )
                
                if success:
                    # tvec (Translation Vector), kameranın eski konuma göre X,Y,Z eksenlerinde metre cinsinden ne kadar kaydığını verir.
                    # Hız (V) = Yerdeğiştirme (tvec) / Zaman (dt)
                    translation_magnitude = np.linalg.norm(tvec)
                    est_speed = translation_magnitude / dt
                    
                    # RANSAC'ın sağlam (inlier) kabul ettiği nokta sayısı
                    inlier_count = len(inliers) if inliers is not None else 0

                    # --- EKRANA BAS ---
                    cv.putText(vis_frame, f"Tahmin Edilen Hiz: {est_speed:.2f} m/s", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv.putText(vis_frame, f"Gercek Hiz (GT)  : {true_speed:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    error = abs(est_speed - true_speed)
                    cv.putText(vis_frame, f"HATA: {error:.2f} m/s", (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv.putText(vis_frame, f"PnP Inliers: {inlier_count}/{len(obj_pts_3d)}", (20, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 5. Bir sonraki kare için hazırlık yap
            old_left = curr_left.copy()
            # Yeni karenin derinlik haritasını çıkar
            left_disp = left_matcher.compute(curr_left, curr_right)
            right_disp = right_matcher.compute(curr_right, curr_left)
            old_disp = wls_filter.filter(left_disp, curr_left, None, right_disp).astype(np.float32) / 16.0
            
            old_p0 = cv.goodFeaturesToTrack(curr_left, mask=None, maxCorners=150, qualityLevel=0.1, minDistance=7)

        cv.imshow('TUSAS - Stereo VIO (Metric Velocity)', vis_frame)

        if cv.waitKey(30) & 0xff == 27: break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
