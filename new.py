import cv2
import numpy as np
import os

# 1. DOSYA KONTROLÜ (Kendi dosya yolunu buraya yaz)
image_path = '1403636579763555584.png' 

if not os.path.exists(image_path):
    print(f"[HATA] Dosya bulunamadı!")
    exit()

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 2. IZGARALAMA (BUCKETING) PARAMETRELERİ
grid_x = 4  # Ekranı yatayda 4 parçaya böl
grid_y = 4  # Ekranı dikeyde 4 parçaya böl
features_per_cell = 30  # Her kutucuktan zorla 30 nokta çıkar (Toplam 16 * 30 = 480 nokta)

h, w = img.shape
cell_h, cell_w = h // grid_y, w // grid_x

all_corners = []

print(f"[SİSTEM] {grid_x}x{grid_y} Izgaralama devrede. Işık patlamaları bypass ediliyor...")

# 3. HER BİR HÜCRE İÇİN SHI-TOMASI ÇALIŞTIR
for i in range(grid_y):
    for j in range(grid_x):
        # Hücrenin sınırlarını belirle
        y1, y2 = i * cell_h, (i + 1) * cell_h
        x1, x2 = j * cell_w, (j + 1) * cell_w
        
        # Resmin sadece o hücresini kes (ROI)
        roi = img[y1:y2, x1:x2]
        
        # Sadece o hücre içinde köşe ara
        corners = cv2.goodFeaturesToTrack(roi, maxCorners=features_per_cell, qualityLevel=0.01, minDistance=5)
        
        # Bulunan köşelerin koordinatlarını orijinal resim boyutuna göre kaydır (offset)
        if corners is not None:
            corners[:, 0, 0] += x1
            corners[:, 0, 1] += y1
            all_corners.extend(corners)

# 4. GÖRSELLEŞTİRME
if len(all_corners) > 0:
    all_corners = np.int0(all_corners)
    print(f"[BAŞARI] Ekranın her yerine yayılmış toplam {len(all_corners)} adet özellik bulundu.")
    
    # Izgara çizgilerini çiz (Sistemin dünyayı nasıl böldüğünü gör)
    for i in range(1, grid_y):
        cv2.line(img_display, (0, i * cell_h), (w, i * cell_h), (255, 0, 0), 1)
    for j in range(1, grid_x):
        cv2.line(img_display, (j * cell_w, 0), (j * cell_w, h), (255, 0, 0), 1)

    # Noktaları çiz
    for pt in all_corners:
        x, y = pt.ravel()
        cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
else:
    print("[HATA] Köşe bulunamadı.")

cv2.imshow('EuRoC - Homojen Dagilmis Referans Noktalari', img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
