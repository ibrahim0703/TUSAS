import cv2
import numpy as np

class RawStereoTracker:
    def __init__(self):
        # Parametreler
        self.feature_params = dict(maxCorners=200, qualityLevel=0.05, minDistance=10, blockSize=3)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Orijinal Matrisler
        self.K_left = np.array([[190.978, 0.0, 254.931], [0.0, 190.973, 256.897], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_left = np.array([0.003482, 0.000715, -0.002053, 0.0002029], dtype=np.float64)
        self.K_right = np.array([[190.442, 0.0, 252.597], [0.0, 190.434, 254.917], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_right = np.array([0.003400, 0.001766, -0.002663, 0.0003299], dtype=np.float64)
        self.R = np.array([[ 0.99999719, 0.00160241, 0.00174676],[-0.00160269, 0.9999987, 0.00016067],[-0.0017465, -0.00016347, 0.99999846]], dtype=np.float64)
        self.T = np.array([-0.10093155, -0.00017163, -0.00067332], dtype=np.float64)

        # SADECE NOKTALARI DÜZELTMEK İÇİN İDEAL PROJEKSİYON MATRİSLERİNİ HESAPLA (Resmi bozmadan)
        self.R1, self.R2, self.P1, self.P2, _ = cv2.fisheye.stereoRectify(
            self.K_left, self.D_left, self.K_right, self.D_right, (512, 512), self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY
        )

    def process_and_get_depth(self, raw_left, raw_right):
        # 1. HAM RESİMDE NOKTA BUL VE TAKİP ET
        p0 = cv2.goodFeaturesToTrack(raw_left, mask=None, **self.feature_params)
        if p0 is None:
            return 0, []

        p1, st, err = cv2.calcOpticalFlowPyrLK(raw_left, raw_right, p0, None, **self.lk_params)

        # LK'nın başarılı bulduğu ham noktaları filtrele
        raw_l_good = p0[st == 1].reshape(-1, 1, 2)
        raw_r_good = p1[st == 1].reshape(-1, 1, 2)

        if len(raw_l_good) == 0:
            return 0, []

        # 2. HAM KOORDİNATLARI MATEMATİKSEL OLARAK KUSURSUZ UZAYA TAŞI (Undistort Points)
        # İşte sihir burada. Resmi değil, sadece piksel kordinatlarını (x,y) hizalıyoruz.
        rect_l = cv2.fisheye.undistortPoints(raw_l_good, self.K_left, self.D_left, R=self.R1, P=self.P1)
        rect_r = cv2.fisheye.undistortPoints(raw_r_good, self.K_right, self.D_right, R=self.R2, P=self.P2)

        # 3. HİZALANMIŞ NOKTALARDA EPIPOLAR TEST VE DERİNLİK (Z)
        f_ideal = self.P1[0, 0] # Ortak ideal odak uzaklığı
        B = 0.1009 # Baseline
        
        valid_depths = []
        
        for i in range(len(rect_l)):
            xl, yl = rect_l[i][0]
            xr, yr = rect_r[i][0]

            # Matematiksel uzaya taşıdığımız için artık Y eksenleri eşit olmak ZORUNDA.
            if abs(yl - yr) < 2.0: 
                disp = xl - xr
                if disp > 0:
                    z = (f_ideal * B) / disp
                    valid_depths.append(z)

        return len(valid_depths), valid_depths

# TEST KODU
if __name__ == "__main__":
    tracker = RawStereoTracker()
    
    # DOSYA YOLLARINI YİNE KENDİNE GÖRE AYARLA
    img_left = cv2.imread('SVO_Veri/cam0/data/1520530308199447626.png', cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread('SVO_Veri/cam1/data/1520530308199447626.png', cv2.IMREAD_GRAYSCALE)
    
    match_count, depths = tracker.process_and_get_depth(img_left, img_right)
    
    print(f"Başarılı Epipolar Eşleşme: {match_count}")
    if match_count > 0:
        print(f"İlk 3 noktanın derinliği (Metre): {depths[:3]}")
        print(f"Ortalama Derinlik: {np.mean(depths):.2f} metre")
