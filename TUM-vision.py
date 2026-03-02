import cv2
import numpy as np

class StereoFlowTracker:
    def __init__(self):
        # --- 1. GFTT PARAMETRELERİ (Sol Kameradan Nokta Çıkarmak İçin) ---
        # maxCorners: En fazla 200 nokta. Fazlası işlemciyi kilitler, azı geometrik hesabı zayıflatır.
        # qualityLevel: 0.01. Kalitesi en yüksek noktanın %1'inden daha kötü olanları çöpe atar. (Gürültü engeller)
        # minDistance: 15. İki nokta birbirine 15 pikselden daha yakın Olamaz! Bu, noktaların bir yere yığılmasını engeller, ekranı tarar.
        self.feature_params = dict(maxCorners=200,
                                   qualityLevel=0.01,
                                   minDistance=15,
                                   blockSize=3)
        
        # --- 2. LUCAS-KANADE (OPTİK AKIŞ) PARAMETRELERİ (Sağ Kamerada Noktayı Bulmak İçin) ---
        # winSize: (15, 15). Algoritma sol resimdeki noktayı, sağ resimde 15x15'lik bir pencere içinde arar.
        # maxLevel: 2. Görüntü piramidi. Hızlı hareketlerde pikselleri kaçırmamak için resmi küçülterek de bakar.
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def get_stereo_matches(self, img_left_rect, img_right_rect):
        """
        Sol ve Sağ HİZALANMIŞ resimleri alır. Sol resimdeki köşeleri bulup, Sağ resimde nereye gittiklerini söyler.
        """
        # Adım 1: Referans noktalarını sadece Sol kameradan çıkar
        # Dönen p0_left boyutu: (N, 1, 2)
        p0_left = cv2.goodFeaturesToTrack(img_left_rect, mask=None, **self.feature_params)
        
        # Eğer duvara çok yakınsak ve hiç köşe yoksa sistemi çökertmemek için kontrol
        if p0_left is None:
            return np.array([]), np.array([])

        # Adım 2: Sol kameradaki noktaları (p0_left), Sağ kamerada Optik Akış ile bul
        # p1_right: Sağ resimdeki yeni konumlar.
        # status (st): Nokta bulunduysa 1, bulunamadıysa 0.
        p1_right, st, err = cv2.calcOpticalFlowPyrLK(img_left_rect, img_right_rect, p0_left, None, **self.lk_params)

        # Adım 3: Sıkı Filtreleme (Sadece Başarılı ve Mantıklı Noktaları Al)
        good_left = []
        good_right = []
        
        for i, (right_pt, left_pt) in enumerate(zip(p1_right, p0_left)):
            if st[i] == 1: # Optik Akış "noktayı buldum" diyorsa...
                xl, yl = left_pt.ravel() # ravel(), o gereksiz (N, 1, 2) boyutunu düz (x,y) yapar
                xr, yr = right_pt.ravel()
                
                # ADIM 4: EPIPOLAR KONTROL (En Kritik Aşama)
                # Aşama 1'de görüntüleri yatayda kusursuz hizaladık. 
                # O zaman Sol kameradaki Y değeri ile Sağ kameradaki Y değeri milimetrik AYNI olmalıdır.
                # Eğer 1 pikselden fazla sapma varsa, algoritma yanlış noktayı bulmuştur! Acımasızca atıyoruz.
                if abs(yl - yr) < 1.0: 
                    good_left.append((xl, yl))
                    good_right.append((xr, yr))

        # Hesaplama kolaylığı için Numpy dizisine çevirip geri döndür
        return np.float32(good_left), np.float32(good_right)
