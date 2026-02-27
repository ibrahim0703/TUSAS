import cv2
import numpy as np
import os

image_path = '1403636579763555584.png' 

if not os.path.exists(image_path):
    print(f"[HATA] Dosya bulunamadı!")
    exit()

img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. CLAHE İLE GÖRÜNTÜYÜ PATLAT
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_raw)
img_display = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

# 2. FAST ALGORİTMASI (Drone'ların gerçek silahı)
# threshold=15: Piksel kontrast farkı. Sayı küçüldükçe daha agresif köşe bulur (10-20 arası idealdir).
fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)

# 3. IZGARALAMA (BUCKETING)
grid_x = 4  
grid_y = 4  
features_per_cell = 30  # Her kutudan zorla 30 nokta

h, w = img_clahe.shape
cell_h, cell_w = h // grid_y, w // grid_x
all_corners = []

print(f"[SİSTEM] FAST Algoritması Devrede. {grid_x}x{grid_y} Izgaralardan özellikler sökülüyor...")

# 4. HER HÜCRE İÇİN FAST ÇALIŞTIR
for i in range(grid_y):
    for j in range(grid_x):
        y1, y2 = i * cell_h, (i + 1) * cell_h
        x1, x2 = j * cell_w, (j + 1) * cell_w
        
        roi = img_clahe[y1:y2, x1:x2]
        
        # FAST ile noktaları bul (KeyPoint nesnesi döner)
        kp = fast.detect(roi, None)
        
        if len(kp) > 0:
            # Noktaları FAST'in kalite skoruna (response) göre en iyiden kötüye sırala
            kp = sorted(kp, key=lambda x: x.response, reverse=True)
            # Sadece en iyi 30 tanesini al (Kutuyu noktaya boğmamak için)
            kp = kp[:features_per_cell]
            
            for p in kp:
                # Koordinatları orijinal resim ofsetine göre kaydır
                x = int(p.pt[0] + x1)
                y = int(p.pt[1] + y1)
                all_corners.append((x, y))

# 5. GÖRSELLEŞTİRME
if len(all_corners) > 0:
    print(f"[BAŞARI] Ekranın her yerine yayılmış toplam {len(all_corners)} adet özellik bulundu.")
    
    # Izgaraları çiz
    for i in range(1, grid_y):
        cv2.line(img_display, (0, i * cell_h), (w, i * cell_h), (255, 0, 0), 1)
    for j in range(1, grid_x):
        cv2.line(img_display, (j * cell_w, 0), (j * cell_w, h), (255, 0, 0), 1)

    # Noktaları çiz
    for (x, y) in all_corners:
        cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)
else:
    print("[HATA] Köşe bulunamadı.")

cv2.imshow('EuRoC - FAST ile Homojen Feature Extraction', img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
