import cv2
import numpy as np

class RawStereoTracker:
    def __init__(self):
        # Orijinal TUM-VI Matrisleri (Fiziksel Gerçeklik)
        self.K_left = np.array([[190.978, 0.0, 254.931], [0.0, 190.973, 256.897], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_left = np.array([0.003482, 0.000715, -0.002053, 0.0002029], dtype=np.float64)
        self.K_right = np.array([[190.442, 0.0, 252.597], [0.0, 190.434, 254.917], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_right = np.array([0.003400, 0.001766, -0.002663, 0.0003299], dtype=np.float64)
        self.R = np.array([[ 0.99999719, 0.00160241, 0.00174676],[-0.00160269, 0.9999987, 0.00016067],[-0.0017465, -0.00016347, 0.99999846]], dtype=np.float64)
        self.T = np.array([-0.10093155, -0.00017163, -0.00067332], dtype=np.float64)

        # Sadece koordinatları hizalamak için projeksiyon matrisleri
        self.R1, self.R2, self.P1, self.P2, _ = cv2.fisheye.stereoRectify(
            self.K_left, self.D_left, self.K_right, self.D_right, (512, 512), self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY
        )
        
        # YENİ SİLAH: ORB (Arama penceresi sınırı yok!)
        self.orb = cv2.ORB_create(nfeatures=500)
        # BFMatcher: Çapraz kontrol (crossCheck) ile yalan eşleşmeleri acımasızca eler
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def process_and_get_depth(self, raw_left, raw_right):
        # 1. ORB ile Nokta ve Kimlik Kartı (Descriptor) Çıkar
        kp1, des1 = self.orb.detectAndCompute(raw_left, None)
        kp2, des2 = self.orb.detectAndCompute(raw_right, None)

        if des1 is None or des2 is None:
            return 0, []

        # 2. Brute-Force ile Eşleştir (Mesafe/Pencere kısıtlaması YOK)
        matches = self.bf.match(des1, des2)
        
        if len(matches) == 0:
            return 0, []

        # Sadece iyi eşleşen piksellerin ham (x,y) koordinatlarını al
        raw_l_good = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        raw_r_good = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 3. HAM KOORDİNATLARI MATEMATİKSEL OLARAK KUSURSUZ UZAYA TAŞI (Görüntüyü bozmadan)
        rect_l = cv2.fisheye.undistortPoints(raw_l_good, self.K_left, self.D_left, R=self.R1, P=self.P1)
        rect_r = cv2.fisheye.undistortPoints(raw_r_good, self.K_right, self.D_right, R=self.R2, P=self.P2)

        # 4. EPIPOLAR TEST VE DERİNLİK (Z) HESABI
        f_ideal = self.P1[0, 0] 
        B = 0.1009 # Baseline
        valid_depths = []
        
        for i in range(len(rect_l)):
            xl, yl = rect_l[i][0]
            xr, yr = rect_r[i][0]

            # ORB pikselleri GFTT kadar keskin alt-piksel (sub-pixel) hassasiyetinde değildir.
            # Bu yüzden Y ekseni toleransını 3.0 piksele çıkardık.
            if abs(yl - yr) < 3.0: 
                disp = xl - xr
                if disp > 0: # Fiziksel olarak sol X, sağ X'ten büyük olmak zorundadır.
                    z = (f_ideal * B) / disp
                    valid_depths.append(z)

        return len(valid_depths), valid_depths

# TEST KODU
if __name__ == "__main__":
    tracker = RawStereoTracker()
    
    # DOSYA YOLLARINI DÜZENLE
    img_left = cv2.imread('SVO_Veri/cam0/data/1520530308199447626.png', cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread('SVO_Veri/cam1/data/1520530308199447626.png', cv2.IMREAD_GRAYSCALE)
    
    match_count, depths = tracker.process_and_get_depth(img_left, img_right)
    
    print(f"[SONUÇ] Başarılı ORB Epipolar Eşleşme: {match_count}")
    if match_count > 0:
        print(f"[MATEMATİK] İlk 3 noktanın derinliği (Metre): {depths[:3]}")
        print(f"[MATEMATİK] Ortalama Derinlik: {np.mean(depths):.2f} metre")
