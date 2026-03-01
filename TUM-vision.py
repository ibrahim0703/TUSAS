import cv2
import numpy as np
import os

# --- KALİBRASYON DEĞERLERİ (TUM-VI Room 1) ---
K_left = np.array([[190.978, 0.0, 254.931],
                   [0.0, 190.973, 256.897],
                   [0.0, 0.0, 1.0]])
D_left = np.array([0.003482, 0.000715, -0.002053, 0.0002029])

K_right = np.array([[190.442, 0.0, 252.597],
                    [0.0, 190.434, 254.917],
                    [0.0, 0.0, 1.0]])
D_right = np.array([0.003400, 0.001766, -0.002663, 0.0003299])

# --- GÖRÜNTÜ YOLLARI (Kendi klasörüne göre ayarla) ---
# mav0/cam0/data ve mav0/cam1/data içindeki AYNI ZAMAN DAMGALI iki resmi seç
left_img_path = './tum_vi_data/room1/mav0/cam0/data/1520530308199447626.png' 
right_img_path = './tum_vi_data/room1/mav0/cam1/data/1520530308199447626.png'

def test_undistort():
    # 1. Ham resimleri oku
    img_left_raw = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    img_right_raw = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    
    if img_left_raw is None or img_right_raw is None:
        print("[HATA] Resimler bulunamadı. Klasör yolunu kontrol et.")
        return

    # 2. Balıkgözü düzeltmesi yap
    # cv2.fisheye.undistortImage kullanıyoruz, çünkü lens tipi 'equidistant'
    img_left_clean = cv2.fisheye.undistortImage(img_left_raw, K_left, D_left, Knew=K_left)
    img_right_clean = cv2.fisheye.undistortImage(img_right_raw, K_right, D_right, Knew=K_right)

    # 3. Görsel Karşılaştırma için yan yana koy
    comparison_left = np.hstack((img_left_raw, img_left_clean))
    
    cv2.imshow("Sol: HAM vs DUZELTILMIS", comparison_left)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_undistort()
