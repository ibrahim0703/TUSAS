import numpy as np
import cv2 as cv

# --- TUSAŞ PROJESİ AYARLARI ---
dosya_adi = 'slow_traffic_small.mp4' 
# NOT: Bu videoda kamera SABİT, arabalar hareketli. 
# O yüzden kodumuzun "0" hız bulması lazım (Doğrusu bu).
# Eğer hareketli bir video (dashcam) koyarsan, gerçek hızı bulur.

cap = cv.VideoCapture(dosya_adi)

# Parametreler (Aynen koruyoruz)
feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
lk_params = dict( winSize  = (15, 15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# İlk kareyi oku
ret, old_frame = cap.read()
if not ret: exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# --- FİZİKSEL VARSAYIMLAR (Piksel -> Metre için) ---
# TUSAŞ İHA'sı için örnek değerler:
FOCAL_LENGTH = 700.0  # Kameranın odak uzaklığı (Piksel cinsinden)
HEIGHT_Z = 10.0       # Yerden yükseklik (Metre) - İHA 10 metrede uçuyor
FPS = 30.0            # Kameranın hızı
DT = 1 / FPS          # Zaman adımı (saniye)

while(1):
    ret, frame = cap.read()
    if not ret: break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 1. Optik Akış Hesapla
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 2. İyi noktaları seç
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # --- MÜHENDİSLİK KISMI BURADA BAŞLIYOR ---
        
        # Her bir noktanın ne kadar kaydığını (dx, dy) hesapla
        # dx: Yatay hareket, dy: Dikey hareket
        displacement = good_new - good_old 
        dx_list = displacement[:, 0]
        dy_list = displacement[:, 1]
        
        # 3. OUTLIER TEMİZLİĞİ (Gürültüyü At)
        # Ortalama (Mean) alma! Çünkü ekrandan geçen tek bir hızlı araba ortalamayı bozar.
        # Medyan (Ortanca) al. Medyan, arka planın (çoğunluğun) hareketini verir.
        median_dx = np.median(dx_list)
        median_dy = np.median(dy_list)
        
        # 4. PİKSELDEN METREYE DÖNÜŞÜM (Geometry)
        # Formül: V = (Piksel_Hızı * Yükseklik) / Odak_Uzaklığı
        # Piksel Hızı = displacement / dt
        
        vx_metric = (median_dx * HEIGHT_Z) / (FOCAL_LENGTH * DT)
        vy_metric = (median_dy * HEIGHT_Z) / (FOCAL_LENGTH * DT)
        
        # TUSAŞ formatında çıktı ver (Konsola bak)
        print(f"Piksel Kayması: ({median_dx:.2f}, {median_dy:.2f}) px | Tahmini Hız: ({vx_metric:.2f}, {vy_metric:.2f}) m/s")

        # Görsellik (Yine de ne olduğunu görelim)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    cv.imshow('TUSAS Vision Speedometer', frame)
    
    k = cv.waitKey(30) & 0xff
    if k == 27: break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
cap.release()
