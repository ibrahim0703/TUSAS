    def get_stereo_matches(self, img_left_rect, img_right_rect):
        # 1. Sol kameradan nokta çıkar
        p0_left = cv2.goodFeaturesToTrack(img_left_rect, mask=None, **self.feature_params)
        
        if p0_left is None:
            print("[DEBUG] KRİTİK HATA: Sol resimde TAKİP EDİLECEK HİÇBİR NOKTA BULUNAMADI!")
            return np.array([]), np.array([])
            
        print(f"[DEBUG] 1. Aşama: Sol resimde {len(p0_left)} adet başlangıç noktası bulundu.")

        # 2. Optik Akış (Lucas-Kanade)
        p1_right, st, err = cv2.calcOpticalFlowPyrLK(img_left_rect, img_right_rect, p0_left, None, **self.lk_params)
        
        basarili_lk_sayisi = np.sum(st == 1)
        print(f"[DEBUG] 2. Aşama: Optik Akış bu noktaların {basarili_lk_sayisi} tanesini sağ resimde bulduğunu iddia ediyor.")

        # 3. Epipolar Filtre
        good_left = []
        good_right = []
        y_farklari = [] # Y eksenindeki sapmaları kaydedelim
        
        for i, (right_pt, left_pt) in enumerate(zip(p1_right, p0_left)):
            if st[i] == 1:
                xl, yl = left_pt.ravel()
                xr, yr = right_pt.ravel()
                
                sapma = abs(yl - yr)
                y_farklari.append(sapma)
                
                # FİLTREYİ GEÇİCİ OLARAK 50 PİKSELE ÇIKARDIK! (Sadece görmek için)
                if sapma < 50.0: 
                    good_left.append((xl, yl))
                    good_right.append((xr, yr))

        ortalama_sapma = np.mean(y_farklari) if len(y_farklari) > 0 else 0
        max_sapma = np.max(y_farklari) if len(y_farklari) > 0 else 0
        
        print(f"[DEBUG] 3. Aşama: Y Eksenindeki Ortalama Sapma: {ortalama_sapma:.2f} piksel")
        print(f"[DEBUG] 3. Aşama: Y Eksenindeki Maksimum Sapma: {max_sapma:.2f} piksel")
        print(f"[DEBUG] SONUÇ: Eşleşen nokta sayısı: {len(good_left)}")

        return np.float32(good_left), np.float32(good_right)
