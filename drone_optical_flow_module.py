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
    
    'FOCAL_LENGTH_PX': 458.654,
    'BASELINE_M': 0.11,  # EuRoC kameraları arasındaki fiziksel mesafe (11 cm)
    
    'LK_PARAMS': dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
}

# =============================================================================
# 2. STEREO DATA LOADER
# =============================================================================
class StereoDataLoader:
    def __init__(self):
        df0 = pd.read_csv(CONFIG['CAM0_CSV'], names=['timestamp', 'filename'], header=0)
        df1 = pd.read_csv(CONFIG['CAM1_CSV'], names=['timestamp', 'filename'], header=0)
        
        df0 = df0.sort_values('timestamp')
        df1 = df1.sort_values('timestamp')
        
        self.stereo_df = pd.merge_asof(df0, df1, on='timestamp', direction='nearest', suffixes=('_left', '_right'))
        self.current_idx = 0
        self.total_frames = len(self.stereo_df)

    def get_stereo_pair(self):
        if self.current_idx >= self.total_frames:
            return None, None
            
        row = self.stereo_df.iloc[self.current_idx]
        img_left = cv.imread(os.path.join(CONFIG['CAM0_DIR'], str(row['filename_left'])), cv.IMREAD_GRAYSCALE)
        img_right = cv.imread(os.path.join(CONFIG['CAM1_DIR'], str(row['filename_right'])), cv.IMREAD_GRAYSCALE)
        
        self.current_idx += 1
        return img_left, img_right

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
# 4. ANA DÖNGÜ (Optik Akış + 3D Derinlik Çıkarımı)
# =============================================================================
def main():
    loader = StereoDataLoader()
    left_matcher, right_matcher, wls_filter = init_wls_stereo_matcher()
    
    # İlk kareyi al
    old_left, _ = loader.get_stereo_pair()
    if old_left is None: return
    
    # Eskiden ızgara çiziyordun. Şimdi SADECE dokulu (köşeli) yerleri bulacağız ki derinlikleri %100 çıksın.
    p0 = cv.goodFeaturesToTrack(old_left, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    while True:
        img_left, img_right = loader.get_stereo_pair()
        if img_left is None: break
            
        # 1. Derinlik Haritasını Çıkar
        left_disp = left_matcher.compute(img_left, img_right)
        right_disp = right_matcher.compute(img_right, img_left)
        filtered_disp = wls_filter.filter(left_disp, img_left, None, right_disp)
        
        # Gerçek piksel kaymasına (Disparity) çevir
        disp_float = filtered_disp.astype(np.float32) / 16.0
        
        vis_frame = cv.cvtColor(img_left, cv.COLOR_GRAY2BGR)

        # 2. Optik Akış ile Noktaları Takip Et
        if p0 is not None and len(p0) > 10:
            p1, st, err = cv.calcOpticalFlowPyrLK(old_left, img_left, p0, None, **CONFIG['LK_PARAMS'])
            
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            valid_3d_points = 0
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                x_px, y_px = int(new[0]), int(new[1])
                
                # Sınır kontrolü (Matris dışına çıkmamak için)
                if 0 <= y_px < disp_float.shape[0] and 0 <= x_px < disp_float.shape[1]:
                    
                    # --- İŞTE O SİHİRLİ MATEMATİK BURADA ---
                    d = disp_float[y_px, x_px]
                    
                    # Sadece geçerli derinliği olan (siyah delik olmayan) noktaları işle
                    if d > 0:
                        # Z = (f * B) / d
                        z_meters = (CONFIG['FOCAL_LENGTH_PX'] * CONFIG['BASELINE_M']) / d
                        
                        # 10 metreden uzak veriler genelde gürültülüdür, filtrele
                        if z_meters < 10.0:
                            valid_3d_points += 1
                            
                            # Görselleştirme: Yakın noktaları Kırmızı, Uzak noktaları Mavi çiz
                            color = (255, max(0, 255 - int(z_meters*25)), 0) 
                            cv.circle(vis_frame, (x_px, y_px), 4, color, -1)
                            # Yanına uzaklığı metre cinsinden yaz
                            cv.putText(vis_frame, f"{z_meters:.1f}m", (x_px+5, y_px-5), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Ekrana Bilgi Bas
            cv.putText(vis_frame, f"Takip Edilen Nokta: {len(good_new)}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.putText(vis_frame, f"Geçerli 3D Nokta: {valid_3d_points}", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            old_left = img_left.copy()
            p0 = good_new.reshape(-1, 1, 2)
            
            # Eğer nokta sayısı çok azaldıysa yeniden özellik bul
            if len(good_new) < 30:
                p0 = cv.goodFeaturesToTrack(img_left, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        else:
            p0 = cv.goodFeaturesToTrack(img_left, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            old_left = img_left.copy()

        cv.imshow('TUSAS - Stereo VIO (3D Point Tracking)', vis_frame)

        if cv.waitKey(30) & 0xff == 27: break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
