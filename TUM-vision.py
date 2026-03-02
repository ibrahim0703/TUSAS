import cv2
import numpy as np

# =========================================================================
# AŞAMA 1: FİZİKSEL HİZALAMA (YIRTILMALAR GİDERİLDİ)
# =========================================================================
class StereoRectifier:
    def __init__(self):
        self.image_size = (512, 512)
        
        # Orijinal Balıkgözü Matrisleri (float64 Zırhı ile)
        self.K_left = np.array([[190.978477, 0.0, 254.931706], [0.0, 190.973307, 256.897442], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_left = np.array([0.003482, 0.000715, -0.002053, 0.0002029], dtype=np.float64)
        
        self.K_right = np.array([[190.442369, 0.0, 252.597253], [0.0, 190.434438, 254.917230], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_right = np.array([0.003400, 0.001766, -0.002663, 0.0003299], dtype=np.float64)

        # Görüntü yırtılmasını engelleyen yeni odak uzaklığı
        self.f_new = 220.0
        self.K_new = np.array([[self.f_new, 0.0, 256.0], 
                               [0.0, self.f_new, 256.0], 
                               [0.0, 0.0, 1.0]], dtype=np.float64)

        # Kameraları birbirine göre döndürmüyoruz (np.eye(3)), sadece Undistort yapıyoruz
        self.map1_l, self.map2_l = cv2.fisheye.initUndistortRectifyMap(
            self.K_left, self.D_left, np.eye(3), self.K_new, self.image_size, cv2.CV_16SC2)
        
        self.map1_r, self.map2_r = cv2.fisheye.initUndistortRectifyMap(
            self.K_right, self.D_right, np.eye(3), self.K_new, self.image_size, cv2.CV_16SC2)

    def process(self, img_left, img_right):
        clean_left = cv2.remap(img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        clean_right = cv2.remap(img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return clean_left, clean_right


# =========================================================================
# AŞAMA 2: NOKTA TAKİBİ VE OPTİK AKIŞ
# =========================================================================
class StereoFlowTracker:
    def __init__(self):
        # Yüksek kalite (0.1) ile sadece en keskin köşeler
        self.feature_params = dict(maxCorners=200, qualityLevel=0.1, minDistance=15, blockSize=3)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def get_stereo_matches(self, img_left_rect, img_right_rect):
        p0_left = cv2.goodFeaturesToTrack(img_left_rect, mask=None, **self.feature_params)
        
        if p0_left is None:
            return np.array([]), np.array([])

        p1_right, st, err = cv2.calcOpticalFlowPyrLK(img_left_rect, img_right_rect, p0_left, None, **self.lk_params)

        good_left = []
        good_right = []
        
        for i, (right_pt, left_pt) in enumerate(zip(p1_right, p0_left)):
            if st[i] == 1:
                xl, yl = left_pt.ravel()
                xr, yr = right_pt.ravel()
                
                # TOLERANS: 3 PİKSEL (Fiziksel montaj ufak sapmalara neden olabilir)
                if abs(yl - yr) < 3.0: 
                    good_left.append((xl, yl))
                    good_right.append((xr, yr))

        return np.float32(good_left), np.float32(good_right)


# =========================================================================
# ANA ÇALIŞTIRMA VE GÖRSELLEŞTİRME
# =========================================================================
def test_and_visualize_pipeline():
    rectifier = StereoRectifier()
    tracker = StereoFlowTracker()

    # !!! DİKKAT: DOSYA YOLLARINI KENDİ BİLGİSAYARINA GÖRE GÜNCELLE !!!
    left_path = 'SVO_Veri/cam0/data/1520530308199447626.png'
    right_path = 'SVO_Veri/cam1/data/1520530308199447626.png'

    img_left_raw = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img_right_raw = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if img_left_raw is None or img_right_raw is None:
        print("[HATA] Görüntüler okunamadı! Dosya yollarını kontrol et.")
        return

    # İşlemler
    img_left_rect, img_right_rect = rectifier.process(img_left_raw, img_right_raw)
    good_left, good_right = tracker.get_stereo_matches(img_left_rect, img_right_rect)
    
    print(f"[SONUÇ] Epipolar testten geçen BAŞARILI eşleşme sayısı: {len(good_left)}")

    if len(good_left) == 0:
        print("[HATA] Eşleşen nokta yok. Görselleştirme atlanıyor.")
        return

    # Renkli çizim için BGR formatı ve yan yana birleştirme
    vis_left = cv2.cvtColor(img_left_rect, cv2.COLOR_GRAY2BGR)
    vis_right = cv2.cvtColor(img_right_rect, cv2.COLOR_GRAY2BGR)
    vis_combined = np.hstack((vis_left, vis_right))
    w = vis_left.shape[1] 

    # Çizgileri Çek
    for i in range(len(good_left)):
        pt_l = (int(good_left[i][0]), int(good_left[i][1]))
        pt_r = (int(good_right[i][0]) + w, int(good_right[i][1])) 
        cv2.circle(vis_combined, pt_l, 4, (0, 0, 255), -1)
        cv2.circle(vis_combined, pt_r, 4, (0, 255, 0), -1)
        cv2.line(vis_combined, pt_l, pt_r, (0, 255, 255), 1)

    # MATEMATİKSEL KANIT (DERİNLİK - Z)
    d = good_left[0][0] - good_right[0][0] 
    if d > 0:
        # Eski f(190) YERİNE YENİ f(220) KULLANILIYOR! Baseline = 0.1009 metre.
        z = (220.0 * 0.1009) / d
        print(f"[MATEMATİK] İlk noktanın piksel kayması (Disparity): {d:.2f} piksel")
        print(f"[MATEMATİK] İlk noktanın kameraya uzaklığı (Derinlik): {z:.2f} metre")

    # Ekrana Bas
    cv2.imshow("Kusursuz Stereo - Yirtilma Giderildi", vis_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_and_visualize_pipeline()
