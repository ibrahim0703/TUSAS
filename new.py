import cv2
import numpy as np
import os

image_path = '1403636579763555584.png' 

if not os.path.exists(image_path):
    print(f"[HATA] Dosya bulunamadı!")
    exit()

# 1. HAM RESMİ OKU
img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. CLAHE İLE GÖRÜNTÜYÜ HACKLEME (Kontrastı Zorla Patlat)
# clipLimit: Kontrastın ne kadar agresif artırılacağı (3.0 genelde karanlık depolar için iyidir)
# tileGridSize: Resmi hangi boyutlarda analiz edeceği
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_raw)

# Çizimleri CLAHE uygulanmış ve aydınlatılmış resim üzerinde göstereceğiz ki farkı gör
img_display = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

# 3. IZGARALAMA (BUCKETING)
grid_x = 4  
grid_y = 4  
features_per_cell = 30  

h, w = img_clahe.shape
cell_h, cell_w = h // grid_y, w // grid_x
all_corners = []

print(f"[SİSTEM] CLAHE Aktif. Karanlık bölgeler aydınlatıldı. {grid_x}x{grid_y} Izgaralama başlatılıyor...")

# 4. HER BİR HÜCRE İÇİN SHI-TOMASI
for i in range(grid_y):
    for j in range(grid_x):
        y1, y2 = i * cell_h, (i + 1) * cell_h
        x1, x2 = j * cell_w, (j + 1) * cell_w
        
        # DİKKAT: Artık karanlık resme değil, CLAHE ile aydınlatılmış resme bakıyoruz!
        roi = img_clahe[y1:y2, x1:x2]
        
        # Kontrastı zaten patlattığımız için qualityLevel'i normal bir seviyede (0.01) tutabiliriz
        corners = cv2.goodFeaturesToTrack(roi, maxCorners=features_per_cell, qualityLevel=0.01, minDistance=5)
        
        if corners is not None:
            corners[:, 0, 0] += x1
            corners[:, 0, 1] += y1
            all_corners.extend(corners)

# 5. GÖRSELLEŞTİRME
if len(all_corners) > 0:
    all_corners = np.int0(all_corners)
    print(f"[BAŞARI] Ekranın her yerine yayılmış toplam {len(all_corners)} adet özellik bulundu.")
    
    for i in range(1, grid_y):
        cv2.line(img_display, (0, i * cell_h), (w, i * cell_h), (255, 0, 0), 1)
    for j in range(1, grid_x):
        cv2.line(img_display, (j * cell_w, 0), (j * cell_w, h), (255, 0, 0), 1)

    for pt in all_corners:
        x, y = pt.ravel()
        cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
else:
    print("[HATA] Köşe bulunamadı.")

cv2.imshow('EuRoC - CLAHE ile Guclendirilmis Feature Extraction', img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
