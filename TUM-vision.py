import cv2
import numpy as np

class RawStereoTracker:
    def __init__(self):
        # 1. ORB BAŞLATICI: GFTT'yi sildik çünkü bize sadece köşe değil, "Kimlik Kartı" (Descriptor) lazım.
        # nfeatures=500: Görüntüden en güçlü 500 noktayı bulmasını istiyoruz.
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # 2. EŞLEŞTİRİCİ (Matcher): Optik Akış'ı (LK) sildik çünkü LK sadece 15 piksel uzağa bakabiliyordu.
        # BFMatcher (Brute-Force): Noktanın kimliğini alır, sağ resimdeki TÜM noktalara bakar, en çok benzeyeni seçer.
        # crossCheck=True: Sadece çift taraflı onaylanan (Sol sağa, sağ sola "en iyi sensin" diyorsa) noktaları alır.
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 3. KAMERA FİZİĞİ: TUM-VI'nin kalibrasyon verileri (Hiçbir şeyi değiştirmedik)
        self.K_left = np.array([[190.978, 0.0, 254.931], [0.0, 190.973, 256.897], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_left = np.array([0.003482, 0.000715, -0.002053, 0.0002029], dtype=np.float64)
        self.K_right = np.array([[190.442, 0.0, 252.597], [0.0, 190.434, 254.917], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_right = np.array([0.003400, 0.001766, -0.002663, 0.0003299], dtype=np.float64)
        self.R = np.array([[ 0.99999719, 0.00160241, 0.00174676],[-0.00160269, 0.9999987, 0.00016067],[-0.0017465, -0.00016347, 0.99999846]], dtype=np.float64)
        self.T = np.array([-0.10093155, -0.00017163, -0.00067332], dtype=np.float64)

        # 4. GİZLİ MATEMATİK UZAYI: Görüntüyü BOZMADAN, sadece piksel (x,y) koordinatlarını 
        # düzeltmek için kullanacağımız projeksiyon matrisleri.
        self.R1, self.R2, self.P1, self.P2, _ = cv2.fisheye.stereoRectify(
            self.K_left, self.D_left, self.K_right, self.D_right, (512, 512), self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY
        )

    def process_and_get_depth(self, raw_left, raw_right):
        # ADIM 1: Ham resimlerdeki noktaları (kp) ve kimliklerini (des) bul.
        kp1, des1 = self.orb.detectAndCompute(raw_left, None)
        kp2, des2 = self.orb.detectAndCompute(raw_right, None)

        if des1 is None or des2 is None:
            return [], [], []

        # ADIM 2: Kimlikleri birbiriyle savaştır ve eşleştir (Mesafe sınırı yok!)
        matches = self.bf.match(des1, des2)
        if len(matches) == 0:
            return [], [], []

        # Ham resimdeki (kıvrımlı ve bozuk) koordinatları çek
        raw_l_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        raw_r_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # ADIM 3: MATEMATİKSEL HİZALAMA (Görüntüyü değil, SADECE NOKTALARI düzeltiyoruz)
        # Bu fonksiyon o bükük (balıkgözü) ham koordinatları alır, 3 boyutlu kusursuz bir 
        # sanal düzleme yansıtır.
        rect_l = cv2.fisheye.undistortPoints(raw_l_pts, self.K_left, self.D_left, R=self.R1, P=self.P1)
        rect_r = cv2.fisheye.undistortPoints(raw_r_pts, self.K_right, self.D_right, R=self.R2, P=self.P2)

        # ADIM 4: EPIPOLAR FİLTRE VE DERİNLİK
        f_ideal = self.P1[0, 0] 
        B = 0.1009 
        
        valid_matches = []
        valid_raw_l = []
        valid_raw_r = []
        
        for i in range(len(rect_l)):
            # Kusursuz matematik uzayındaki koordinatlar
            xl, yl = rect_l[i][0]
            xr, yr = rect_r[i][0]

            # Y ekseni farkı (Epipolar kısıtlama). ORB piksel seviyesinde olduğu için 
            # hata payını 3.0 pikselde tuttuk.
            if abs(yl - yr) < 3.0: 
                disp = xl - xr # X eksenindeki kayma (Disparity)
                if disp > 0:
                    z = (f_ideal * B) / disp # Z = (Odak_Uzaklığı * Baseline) / Disparity
                    
                    # Görselleştirme için SADECE matematiksel testi geçen ham noktaları kaydediyoruz
                    valid_raw_l.append(raw_l_pts[i][0])
                    valid_raw_r.append(raw_r_pts[i][0])
                    valid_matches.append(z)

        return valid_raw_l, valid_raw_r, valid_matches

# =========================================================================
# GÖRSELLEŞTİRME MODÜLÜ
# =========================================================================
if __name__ == "__main__":
    tracker = RawStereoTracker()
    
    # DİKKAT: KENDİ DOSYA YOLLARINI YAZ
    img_left = cv2.imread('SVO_Veri/cam0/data/1520530308199447626.png', cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread('SVO_Veri/cam1/data/1520530308199447626.png', cv2.IMREAD_GRAYSCALE)
    
    # Algoritmayı çalıştır
    pts_l, pts_r, depths = tracker.process_and_get_depth(img_left, img_right)
    
    print(f"[SONUÇ] Fiziksel Testten Geçen Başarılı Eşleşme Sayısı: {len(depths)}")
    
    if len(depths) > 0:
        print(f"[MATEMATİK] Ortalama Derinlik: {np.mean(depths):.2f} metre")
        
        # Resimleri yan yana birleştir (Renkli çizim için BGR yapıyoruz)
        vis_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        vis_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
        vis_combined = np.hstack((vis_left, vis_right))
        w = vis_left.shape[1]

        # Çizgileri Çiz
        for i in range(len(pts_l)):
            pt_l = (int(pts_l[i][0]), int(pts_l[i][1]))
            # Sağdaki noktanın X değerine, sol resmin genişliğini (w) ekliyoruz ki sağdaki resmin üstüne düşsün
            pt_r = (int(pts_r[i][0]) + w, int(pts_r[i][1])) 
            
            cv2.circle(vis_combined, pt_l, 4, (0, 0, 255), -1) # Sol nokta Kırmızı
            cv2.circle(vis_combined, pt_r, 4, (0, 255, 0), -1) # Sağ nokta Yeşil
            cv2.line(vis_combined, pt_l, pt_r, (0, 255, 255), 1) # Çizgi Sarı

        # Ekrana bas
        cv2.imshow("Ham Resimde ORB Takibi (Dogru Yontem)", vis_combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[HATA] Geçerli nokta bulunamadı.")
