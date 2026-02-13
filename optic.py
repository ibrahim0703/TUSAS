import numpy as np
import cv2 as cv

# --- AYAR 1: Video Dosyasının Yolu ---
# Video dosyan python kodunla aynı klasörde olsun.
# Buraya videonun tam adını uzantısıyla (.mp4) yaz.
dosya_adi = 'slow_traffic_small.mp4' 
cap = cv.VideoCapture(dosya_adi)

# --- AYAR 2: Köşe Bulma Parametreleri (Shi-Tomasi) ---
# maxCorners: Takip edilecek maksimum nokta sayısı (100 tane)
# qualityLevel: Nokta kalitesi (düşürürsen daha çok ama kötü nokta bulur)
# minDistance: Noktalar arası minimum mesafe (birbirine çok yapışmasınlar)
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# --- AYAR 3: Lucas-Kanade Optik Akış Parametreleri ---
# winSize: Arama penceresi boyutu. Büyük yaparsan hızlı hareketi yakalar ama detay kaybolur.
# maxLevel: Piramit seviyesi (Videoda bahsettiğimiz konu!). 2 demek, görüntüyü 2 kere küçültüp bakar.
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Rastgele renkler oluştur (Çizim için)
color = np.random.randint(0, 255, (100, 3))

# --- ADIM A: İlk Kareyi Al ve Noktaları Bul ---
ret, old_frame = cap.read()
if not ret:
    print("HATA: Video dosyası bulunamadı veya açılamadı!")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0: Takip edilecek ilk noktalar (Köşeler)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Çizim maskesi oluştur
mask = np.zeros_like(old_frame)

print("Video işleniyor... Çıkmak için 'ESC' tuşuna bas.")

while(1):
    ret, frame = cap.read()
    if not ret:
        break # Video bitti

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # --- ADIM B: Optik Akışı Hesapla (Sihir Burada) ---
    # p0: Eski noktalar, p1: Yeni noktalar
    # st: Status (1 ise nokta bulundu, 0 ise kayboldu)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # --- ADIM C: İyi Noktaları Seç ---
    if p1 is not None:
        good_new = p1[st==1] # Takibi süren yeni noktalar
        good_old = p0[st==1] # Bunların eski halleri

    # --- ADIM D: Çizim Yap ---
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel() # Yeni koordinat (x, y)
        c, d = old.ravel() # Eski koordinat (x, y)
        
        # Hareketi çizgi olarak maskeye çiz
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        # Mevcut konuma top koy
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
    img = cv.add(frame, mask)

    cv.imshow('Optical Flow Demo', img)

    # ESC tuşuna basınca çık
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # --- ADIM E: Döngüyü Güncelle ---
    # Şimdiki kare, bir sonraki adım için "Eski Kare" olur
    old_gray = frame_gray.copy()
    # Şimdiki noktalar, bir sonraki adım için "Eski Noktalar" olur
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
cap.release()
