import cv2
import numpy as np
import os
import glob

# ====================================================================
# 1. SİSTEM PARAMETRELERİ VE KALİBRASYON (EuRoC Verileri)
# ====================================================================
# Bu değerler olmadan piksellerin fiziksel bir anlamı yoktur.
f_x, f_y = 458.654, 457.296
c_x, c_y = 367.215, 248.375
baseline = 0.110 # Sol ve sağ kamera arası 11 cm

K = np.array([[f_x,   0, c_x],
              [  0, f_y, c_y],
              [  0,   0,   1]], dtype=np.float64)

dist_coeffs = np.array([-0.2834, 0.0739, 0.00019, 0.000017])

# ====================================================================
# 2. MODÜLLER (Derinlik ve Özellik Çıkarımı)
# ====================================================================
# Gerçek zamanlı sistemlerde SGBM ağır kalabilir, bu yüzden daha hızlı 
# olan StereoBM (Block Matching) kullanıyoruz. (Gerekirse SGBM yapabilirsin)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

def get_features_fast_clahe(img_gray):
    """
    Ekranı ızgaralara bölerek her bölgeden zorla nokta (feature) süzer.
    Karanlık bölgeleri CLAHE ile patlatır.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    grid_x, grid_y = 4, 4
    h, w = img_clahe.shape
    cell_h, cell_w = h // grid_y, w // grid_x
    corners = []
    
    for i in range(grid_y):
        for j in range(grid_x):
            roi = img_clahe[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            kp = fast.detect(roi, None)
            if len(kp) > 0:
                # Her ızgaradan en güçlü 20 noktayı al
                kp = sorted(kp, key=lambda x: x.response, reverse=True)[:20]
                for p in kp:
                    corners.append([[np.float32(p.pt[0] + j*cell_w), np.float32(p.pt[1] + i*cell_h)]])
    return np.array(corners)

def calculate_3d_points(p0_2d, disparity_map):
    """
    SADECE FAST ile bulduğumuz 2D noktaların Derinliğini (Z) hesaplar.
    Bütün resmi hesaplamaktan bizi kurtaran rasyonel mühendislik hamlesidir.
    """
    points_3d = []
    valid_points_2d = []
    
    for pt in p0_2d:
        u, v = int(pt[0][0]), int(pt[0][1])
        # O pikseldeki kaymayı (disparity) okuyoruz. 
        # StereoBM 16 ile çarpılmış değer verir, 16'ya bölerek gerçeği buluruz.
        d = disparity_map[v, u] / 16.0 
        
        # Eğer d çok küçükse (sonsuzluk) veya negatifse, nokta geçersizdir (çöpe at).
        if d > 1.0:
            Z = (f_x * baseline) / d
            X = ((u - c_x) * Z) / f_x
            Y = ((v - c_y) * Z) / f_y
            points_3d.append([X, Y, Z])
            valid_points_2d.append([[np.float32(u), np.float32(v)]])
            
    return np.array(points_3d, dtype=np.float32), np.array(valid_points_2d, dtype=np.float32)

# ====================================================================
# 3. BAŞLANGIÇ (t=0) KURULUMU
# ====================================================================
# KLASÖR YOLUNU KENDİ BİLGİSAYARINA GÖRE GÜNCELLE
left_folder = 'mav0/cam0/data'  
right_folder = 'mav0/cam1/data'

left_images = sorted(glob.glob(os.path.join(left_folder, '*.png')))
right_images = sorted(glob.glob(os.path.join(right_folder, '*.png')))

print("[SİSTEM] Boru hattı başlatılıyor. Geçmiş ile Gelecek bağlanıyor...")

# t=0 anı resimleri
old_left = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
old_right = cv2.imread(right_images[0], cv2.IMREAD_GRAYSCALE)

# 1. Aşama: t=0 anı için Disparity (Derinlik) haritası çıkar
old_disparity = stereo.compute(old_left, old_right)

# 2. Aşama: t=0 anı sol kameradan 2D Özellikleri (Features) bul
p0_2d_raw = get_features_fast_clahe(old_left)

# 3. Aşama: O özelliklerin 3D (X, Y, Z) koordinatlarını bul ve sadece geçerli olanları al
p0_3D, p0_2D = calculate_3d_points(p0_2d_raw, old_disparity)

# Lucas-Kanade Optik Akış Parametreleri
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# ====================================================================
# 4. ZAMAN DÖNGÜSÜ (Video Stream & PnP Hız Tahmini)
# ====================================================================
for i in range(1, len(left_images)):
    # Yeni an (t=1) resimlerini oku
    new_left = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
    new_right = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)
    
    # 4. Aşama: Optik Akış (Zamanı bağla) -> p0 noktaları yeni resimde nereye gitti?
    p1_2D, st, err = cv2.calcOpticalFlowPyrLK(old_left, new_left, p0_2D, None, **lk_params)
    
    # Sadece başarıyla takip edilen noktaları süz (st == 1)
    good_new_2D = p1_2D[st == 1]
    good_old_3D = p0_3D[st.flatten() == 1] # 3D noktalarımızı da aynı filtreyle süzüyoruz!
    
    # 5. Aşama: PnP (Rotasyon ve Ötelemeyi Kır)
    # Eğer elimizde denklem çözecek kadar 3D-2D eşleşmesi kaldıysa:
    if len(good_new_2D) > 10:
        basari, R_vec, t_vec, inliers = cv2.solvePnPRansac(
            objectPoints=good_old_3D, 
            imagePoints=good_new_2D, 
            cameraMatrix=K, 
            distCoeffs=dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if basari:
            # t_vec: Kameranın bir önceki kareye göre Z (İleri), X (Sağ), Y (Aşağı) Hareketi
            x_hareket = t_vec[0][0]
            y_hareket = t_vec[1][0]
            z_hareket = t_vec[2][0] # Drone'un İleri/Geri gidişi (m/s)
            
            # Hızı konsola yazdır
            print(f"Kare {i}: Z-Hızı: {z_hareket*20:.2f} m/s | X-Kayma: {x_hareket*20:.2f} m/s")
            # Neden 20 ile çarptık? Çünkü EuRoC 20 Hz (Saniyede 20 kare) çekiyor. 
            # 1 kareden diğerine olan öteleme, o saniyenin 1/20'si kadardır.
            
    # Görselleştirme (Optik Akış okları)
    frame_color = cv2.cvtColor(new_left, cv2.COLOR_GRAY2BGR)
    for (new, old) in zip(good_new_2D, p0_2D[st == 1]):
        a, b = int(new[0]), int(new[1])
        c, d = int(old[0]), int(old[1])
        cv2.arrowedLine(frame_color, (c, d), (a, b), (0, 255, 0), 2, tipLength=0.3)
    
    cv2.imshow('SVO: Optik Akis & PnP', frame_color)
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
    # --- DÖNGÜYÜ İLERİ SAR (GELECEK, ARTIK GEÇMİŞ OLDU) ---
    old_left = new_left.copy()
    
    # Eğer nokta sayımız çok azaldıysa (drone körleşiyorsa), PnP'yi beslemek için sistemi SIFIRLA
    if len(good_new_2D) < 80:
        print("[UYARI] Noktalar kayboluyor, Sistem Yeniden Kalibre Ediliyor...")
        old_disparity = stereo.compute(old_left, new_right)
        p0_2d_raw = get_features_fast_clahe(old_left)
        p0_3D, p0_2D = calculate_3d_points(p0_2d_raw, old_disparity)
    else:
        # PnP'ye bir sonraki döngüde vermek üzere, şu anki 2D noktalarımızın YENİ 3D derinliklerini hesaplamalıyız!
        old_disparity = stereo.compute(old_left, new_right)
        p0_3D, p0_2D = calculate_3d_points(good_new_2D.reshape(-1, 1, 2), old_disparity)

cv2.destroyAllWindows()
