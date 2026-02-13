import numpy as np
import cv2 as cv

# --- VİDEO DOSYASI ---
dosya_adi = 'araba_videosu.mp4'  # Videonun ismini buraya yaz
cap = cv.VideoCapture(dosya_adi)

# Parametreler (Aynı)
feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
lk_params = dict( winSize  = (15, 15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# İlk kareyi oku ve başlat
ret, old_frame = cap.read()
if not ret: exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# İlk noktaları bul
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Eğer hiç nokta bulamazsa boş bir array ile başlat
if p0 is None:
    p0 = np.empty((0, 1, 2), dtype=np.float32)

# Maske (Görsellik için)
mask = np.zeros_like(old_frame)

# --- FİZİK PARAMETRELERİ (Örnek) ---
# Gerçek değerleri bilmiyoruz, varsayımsal:
FOCAL_LENGTH = 1000.0
HEIGHT_Z = 1.5 # Kamera yerden 1.5 metre yukarıda olsun (Araba için)
DT = 1/30.0    # 30 FPS varsayımı

while(1):
    ret, frame = cap.read()
    if not ret: break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Eğer takip edecek nokta kalmadıysa veya çok azaldıysa YENİDEN BUL
    if len(p0) < 10:  # Eşik değerimiz 10 nokta
        print("--- Noktalar azaldı, yeni noktalar taranıyor... ---")
        new_features = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        
        # Eğer yeni nokta bulduysa eskilere ekle veya yenile
        if new_features is not None:
             # Basitlik için direkt p0'ı yeniliyoruz. 
             # (İleri seviyede eski ve yeniyi birleştirmek gerekir ama bu yeterli)
             p0 = new_features
             old_gray = frame_gray.copy()
             mask = np.zeros_like(old_frame) # Çizgileri temizle ki ekran karışmasın
             continue # Bu kareyi atla, bir sonrakinde hesapla

    # 1. Optik Akış Hesapla
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 2. İyi noktaları seç
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # Nokta sayısı yeterli mi kontrolü
        if len(good_new) > 0:
            # --- HIZ HESABI ---
            displacement = good_new - good_old 
            dx_list = displacement[:, 0]
            dy_list = displacement[:, 1]
            
            # Outlier Temizliği (Medyan)
            median_dx = np.median(dx_list)
            median_dy = np.median(dy_list)
            
            # Piksel -> Metre/Saniye (Basit Dönüşüm)
            # Not: Bu, "Kamera ne kadar döndü/gitti" bilgisidir.
            vx_metric = (median_dx * HEIGHT_Z) / (FOCAL_LENGTH * DT)
            vy_metric = (median_dy * HEIGHT_Z) / (FOCAL_LENGTH * DT)
            
            # Hızı çok küçükse (duruyorsa) gürültüyü sıfırla
            if abs(vx_metric) < 0.1: vx_metric = 0
            if abs(vy_metric) < 0.1: vy_metric = 0

            print(f"Hız Vektörü: X={vx_metric:.2f} m/s | Y={vy_metric:.2f} m/s | Takip Edilen Nokta: {len(good_new)}")

            # --- GÖRSELLİK ---
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                # Yeşil çizgi: Hareket vektörü
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                # Kırmızı nokta: Şu anki konum
                frame = cv.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
            
            img = cv.add(frame, mask)
            cv.imshow('Otonom Hiz Algilama', img)
        
        # Bir sonraki adım için güncelle
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        # Eğer p1 None dönerse (takip koptuysa), p0'ı boşalt ki yukarıdaki "if len < 10" çalışsın
        p0 = np.empty((0, 1, 2), dtype=np.float32)

    k = cv.waitKey(30) & 0xff
    if k == 27: break

cv.destroyAllWindows()
cap.release()
