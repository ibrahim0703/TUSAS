import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ==========================================
# 1. AYARLAR
# ==========================================
video_path = 'araba_videosu.mp4'       # Videonun adı
ground_truth_path = 'hiz_verisi.txt'   # Elindeki 10000x1'lik veri dosyası
GT_FREQ = 20.0                         # Verinin frekansı (Sen 20 Hz dedin)

# Fiziksel Parametreler (Başlangıç Tahmini)
# Not: Kodun sonunda "Önerilen Katsayı"yı görünce burayı güncelleyeceksin.
FOCAL_LENGTH = 900.0 
HEIGHT_Z = 1.5       
PIXEL_TO_METRIC = HEIGHT_Z / (FOCAL_LENGTH * (1/30.0)) # Başlangıç varsayımı

# ==========================================
# 2. GERÇEK VERİYİ YÜKLE
# ==========================================
print("Ground Truth verisi yükleniyor...")
try:
    gt_speed = np.loadtxt(ground_truth_path)
    # Eğer veri tek satırsa veya şekli bozuksa düzelt
    gt_speed = gt_speed.reshape(-1) 
    print(f"-> {len(gt_speed)} adet veri noktası yüklendi.")
except Exception as e:
    print(f"HATA: Veri dosyası okunamadı! ({e})")
    exit()

# Gerçek verinin zaman eksenini oluştur
gt_time = np.arange(len(gt_speed)) / GT_FREQ

# ==========================================
# 3. VİDEO İŞLEME VE HIZ TAHMİNİ
# ==========================================
cap = cv.VideoCapture(video_path)
VIDEO_FPS = cap.get(cv.CAP_PROP_FPS)
if VIDEO_FPS == 0: VIDEO_FPS = 30.0 # Hata olursa varsayılan
DT = 1.0 / VIDEO_FPS

# Dönüşüm katsayısını gerçek FPS ile güncelle
PIXEL_TO_METRIC = HEIGHT_Z / (FOCAL_LENGTH * DT)

print(f"Video Analizi Başlıyor... FPS: {VIDEO_FPS:.2f}")

# Parametreler
feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

ret, old_frame = cap.read()
if not ret: exit()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

est_speed = [] # Tahmin ettiğimiz hızları buraya atacağız

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Nokta Yenileme
    if p0 is None or len(p0) < 50:
        p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()
        est_speed.append(0) # Veri yoksa 0 bas
        frame_idx += 1
        continue

    # Optik Akış
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        displacement = good_new - good_old
        dy = displacement[:, 1]
        
        # --- RANSAC / IQR FİLTRESİ ---
        if len(dy) > 0:
            q75, q25 = np.percentile(dy, [75 ,25])
            iqr = q75 - q25
            lower = q25 - (1.5 * iqr)
            upper = q75 + (1.5 * iqr)
            clean_dy = dy[(dy >= lower) & (dy <= upper)]
            
            if len(clean_dy) > 0:
                px_speed = np.mean(clean_dy)
            else:
                px_speed = 0
        else:
            px_speed = 0
            
        # Hız Dönüşümü (- çünkü dünya aşağı akar)
        # Sadece pozitif (ileri) hızları alalım, geri gitme yok varsayalım
        v_mps = max(0, -px_speed * PIXEL_TO_METRIC)
        est_speed.append(v_mps)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        est_speed.append(0)
        
    frame_idx += 1
    # İlerlemeyi göster
    if frame_idx % 100 == 0: print(f"İşlenen Kare: {frame_idx}")

cap.release()
print("Video bitti.")

# ==========================================
# 4. SENKRONİZASYON VE ANALİZ
# ==========================================
est_speed = np.array(est_speed)
est_time = np.arange(len(est_speed)) / VIDEO_FPS # Videonun zaman ekseni

# Sorun: Video ve Veri süreleri tutmuyor olabilir.
# Çözüm: Video süresi kadar olan kısmı karşılaştıralım.
min_time = min(gt_time[-1], est_time[-1])

# Sadece ortak zaman dilimindeki verileri al
gt_valid_mask = gt_time <= min_time
est_valid_mask = est_time <= min_time

gt_time_crop = gt_time[gt_valid_mask]
gt_speed_crop = gt_speed[gt_valid_mask]

est_time_crop = est_time[est_valid_mask]
est_speed_crop = est_speed[est_valid_mask]

# --- KRİTİK ADIM: Estimasyon verisini GT zamanlarına İnterpole et (Eşitle) ---
# Böylece RMSE hesaplayabiliriz
est_speed_interp = np.interp(gt_time_crop, est_time_crop, est_speed_crop)

# --- KALİBRASYON KATSAYISI HESABI ---
# Bizim verimiz ile gerçek veri arasında sadece "Büyüklük" farkı varsa bunu bulalım.
# Scale Factor = mean(GT) / mean(Est)
if np.mean(est_speed_interp) > 0.1: # Sıfıra bölmeyi engelle
    scale_factor = np.mean(gt_speed_crop) / np.mean(est_speed_interp)
else:
    scale_factor = 1.0

print(f"\n=== SONUÇ RAPORU ===")
print(f"Önerilen Ölçek Çarpanı (Scale Factor): {scale_factor:.4f}")
print(f"-> Eğer bu sayı 1'den çok farklıysa, FOCAL_LENGTH değerini buna göre güncellemelisin.")
print(f"-> Yeni FOCAL_LENGTH = Mevcut ({FOCAL_LENGTH}) / {scale_factor:.4f}")

# Kalibre edilmiş tahmin (Grafik için)
est_speed_calibrated = est_speed_interp * scale_factor

# Hata Hesabı (RMSE)
rmse = np.sqrt(np.mean((gt_speed_crop - est_speed_calibrated)**2))
print(f"Hata (RMSE): {rmse:.2f} m/s (Kalibrasyon sonrası)")

# ==========================================
# 5. GRAFİK ÇİZİMİ
# ==========================================
plt.figure(figsize=(14, 8))

# Üst Grafik: Ham Karşılaştırma
plt.subplot(2, 1, 1)
plt.plot(gt_time_crop, gt_speed_crop, 'r-', label='Ground Truth (Gerçek Veri)')
plt.plot(est_time_crop, est_speed_crop, 'b--', alpha=0.7, label='Senin Kodun (Ham)')
plt.title(f"Ham Veri Karşılaştırması (Scale Factor Öncesi)")
plt.ylabel("Hız (m/s)")
plt.legend()
plt.grid()

# Alt Grafik: Kalibre Edilmiş Karşılaştırma
plt.subplot(2, 1, 2)
plt.plot(gt_time_crop, gt_speed_crop, 'r-', linewidth=2, label='Ground Truth')
plt.plot(gt_time_crop, est_speed_calibrated, 'g-', alpha=0.8, linewidth=1.5, label='Senin Kodun (Kalibre Edilmiş)')
plt.title(f"Kalibre Edilmiş Karşılaştırma (RMSE: {rmse:.2f} m/s)")
plt.xlabel("Zaman (saniye)")
plt.ylabel("Hız (m/s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
