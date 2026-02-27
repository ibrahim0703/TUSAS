import cv2
import numpy as np
import os

# 1. DOSYA KONTROLÜ (EuRoC verisine yönlendir)
# Kendi bilgisayarındaki tam yolu veya dosya adını buraya yaz
image_path = 'Euroc_data_klasorun/mav0/cam0/data/1403636579763555584.png' 

if not os.path.exists(image_path):
    print(f"[HATA] Dosya bulunamadı! Lütfen image_path yolunu doğru ayarla: {image_path}")
    exit()

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 2. AGRESIF SHI-TOMASI PARAMETRELERİ (EuRoC için özel ayar)
max_corners = 500       
# DİKKAT: Kaliteyi 0.01'den 0.001'e düşürdük. En ufak detayı bile yakalayacak.
quality_level = 0.001   
min_distance = 10       

print("[SİSTEM] EuRoC verisi üzerinde agresif özellik çıkarımı (Feature Extraction) başlatıldı...")

# 3. KÖŞELERİ BUL
corners = cv2.goodFeaturesToTrack(img, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)

# 4. GÖRSELLEŞTİRME
if corners is not None:
    corners = np.int0(corners)
    print(f"[BAŞARI] {len(corners)} adet referans noktası (feature) bulundu.")
    
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
else:
    print("[HATA] Hiç köşe bulunamadı. Parametreleri daha da düşürmemiz gerekebilir.")

cv2.imshow('EuRoC - Asama 1: Referans Noktalari', img_display)
print("[BİLGİ] Çıkan yeşil noktaların nerelere tutunduğunu incele.")
cv2.waitKey(0)
cv2.destroyAllWindows()
