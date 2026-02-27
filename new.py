import cv2
import numpy as np
import os
import glob

# 1. KLASÖR YOLU (KENDİ BİLGİSAYARINDAKİ YOLU GÜNCELLE)
# İçinde .png olan veri klasörünü göster (Örn: 'mav0/cam0/data')
folder_path = 'BURAYA_KLASOR_YOLUNU_YAZ' 

if not os.path.exists(folder_path):
    print(f"[HATA] Klasör bulunamadı: {folder_path}")
    exit()

# Resimleri zaman damgasına göre kronolojik sırala
image_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))

if len(image_files) < 2:
    print("[HATA] Klasörde yeterli resim yok!")
    exit()

# 2. LUCAS-KANADE (OPTICAL FLOW) PARAMETRELERİ
lk_params = dict(winSize=(21, 21),  # Algoritma noktanın yeni yerini 21x21 piksellik bir alanda arar
                 maxLevel=3,        # Hızlı hareketleri kaçırmamak için resmi piramit gibi küçültüp arar
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# 3. KÖŞE BULMA FONKSİYONU (Önceki yazdığımız CLAHE + FAST + Bucketing)
def get_features(img_gray):
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    grid_x, grid_y = 4, 4
    features_per_cell = 30
    h, w = img_gray.shape
    cell_h, cell_w = h // grid_y, w // grid_x
    corners = []
    
    for i in range(grid_y):
        for j in range(grid_x):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            roi = img_gray[y1:y2, x1:x2]
            kp = fast.detect(roi, None)
            if len(kp) > 0:
                kp = sorted(kp, key=lambda x: x.response, reverse=True)[:features_per_cell]
                for p in kp:
                    # Optical Flow, numpy float32 tipinde array ister
                    corners.append([[np.float32(p.pt[0] + x1), np.float32(p.pt[1] + y1)]])
    return np.array(corners)

# 4. İLK KAREYİ HAZIRLA
old_frame_raw = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
old_gray = clahe.apply(old_frame_raw)

# İlk noktaları (Feature) bul
p0 = get_features(old_gray)

# Çizgileri (Kuyrukları) çizeceğimiz boş ve şeffaf bir maske oluştur
mask = np.zeros_like(cv2.cvtColor(old_frame_raw, cv2.COLOR_GRAY2BGR))

print(f"[SİSTEM] Optik Akış başlatılıyor. Toplam {len(image_files)} kare işlenecek...")

# 5. ZAMAN DÖNGÜSÜ (Video Stream)
for i in range(1, len(image_files)):
    frame_raw = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
    frame_gray = clahe.apply(frame_raw)
    frame_color = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # p0 noktalarının, yeni karedeki (p1) yerini hesapla
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Sadece başarıyla takip edilen noktaları seç (status == 1)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Hareketi görselleştir (Çizgileri çiz)
    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        
        # Hareket eden rotayı yeşil çizgiyle çiz
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        # Noktanın şu anki konumuna kırmızı daire koy
        frame_color = cv2.circle(frame_color, (a, b), 4, (0, 0, 255), -1)

    # Çizgileri ve gerçek resmi üst üste bindir
    img = cv2.add(frame_color, mask)

    cv2.imshow('EuRoC - Lucas Kanade Optical Flow', img)

    # Saniyede 30 kare gibi oynat. (Çıkmak için ESC'ye bas)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Gelecek döngü için zamanı ileri sar (Yeni kare, artık eski kare oldu)
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # KRİTİK OTONOMİ KURALI: Drone dönerken noktalar ekranın dışına çıkar.
    # Eğer takip edilen nokta sayısı 100'ün altına düşerse, sistemi yeniden besle!
    if len(p0) < 100:
        print("[UYARI] Takip edilen noktalar kayboluyor! Yeni referanslar sökülüyor...")
        p0 = get_features(old_gray)
        mask = np.zeros_like(frame_color) # Eski çizgileri temizle ki ekran çorbaya dönmesin

cv2.destroyAllWindows()
