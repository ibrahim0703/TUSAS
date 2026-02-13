import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ==========================================
# 1. AYARLAR (DOSYA İSİMLERİNİ KONTROL ET)
# ==========================================
video_path = 'araba_videosu.mp4'       # Senin video dosyan
ground_truth_path = 'hiz_verisi.txt'   # Senin hız verisi dosyan
GT_FREQ = 20.0                         # Hız verisinin frekansı (20 Hz demiştin)

# Başlangıç Fiziksel Tahminleri (Kod sonunda otomatik düzelecek)
FOCAL_LENGTH = 1000.0 
HEIGHT_Z = 1.5       
# ==========================================

def get_roi_mask(frame):
    """
    Görüntünün sadece YOL kısmını alan maske.
    Kaputu ve Gökyüzünü kör eder.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # --- MASKE AYARLARI (Gerekirse burayı videona göre oynarsın) ---
    # Yamuk (Trapezoid) şeklinde bir alan belirliyoruz.
    pts = np.array([[
        (int(w * 0.1), int(h * 0.90)),  # Sol Alt (Kaputun hemen üstü)
        (int(w * 0.35), int(h * 0.45)), # Sol Üst (Ufuk çizgisi altı)
        (int(w * 0.65), int(h * 0.45)), # Sağ Üst
        (int(w * 0.9), int(h * 0.90))   # Sağ Alt
    ]], dtype=np.int32)
    
    cv.fillPoly(mask, [pts], 255)
    return mask, pts

# ==========================================
# 2. VERİ YÜKLEME VE HAZIRLIK
# ==========================================
print("Ground Truth verisi yükleniyor...")
try:
    gt_speed = np.loadtxt(ground_truth_path)
    gt_speed = gt_speed.reshape(-1) # Tek boyutlu hale getir
    gt_time = np.arange(len(gt_speed)) / GT_FREQ
    print(f"-> {len(gt_speed)} veri noktası yüklendi.")
except Exception as e:
    print(f"HATA: Veri dosyası okunamadı! {e}")
    exit()

cap = cv.VideoCapture(video_path)
VIDEO_FPS = cap.get(cv.CAP_PROP_FPS)
if VIDEO_FPS == 0: VIDEO_FPS = 30.0
DT = 1.0 / VIDEO_FPS

# İlk Dönüşüm Katsayısı
PIXEL_TO_METRIC = HEIGHT_Z / (FOCAL_LENGTH * DT)

# ==========================================
# 3. ANA DÖNGÜ (OPTICAL FLOW)
# ==========================================
# Parametreler
feature_params = dict(maxCorners=150, qualityLevel=0.2, minDistance=7, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

ret, old_frame = cap.read()
if not ret: exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
roi_mask, roi_pts = get_roi_mask(old_frame) # İLK MASKELEME

# Sadece maskelenmiş alanda nokta bul
p0 = cv.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)

est_speed = []
mask_vis = np.zeros_like(old_frame)

print(f"Analiz Başladı... FPS: {VIDEO_FPS:.2f}")
print("Çıkmak için pencereye tıklayıp 'ESC' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Görselleştirme için temiz kopya
    vis_frame = frame.copy()

    # --- NOKTA YENİLEME (Kritik Bölüm) ---
    # Eğer nokta sayısı azalırsa, yine SADECE ROI İÇİNDE yeni nokta bul
    if p0 is None or len(p0) < 50:
        p0 = cv.goodFeaturesToTrack(frame_gray, mask=roi_mask, **feature_params)
        old_gray = frame_gray.copy()
        est_speed.append(est_speed[-1] if est_speed else 0) # Önceki hızı koru
        continue

    # --- OPTICAL FLOW ---
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # Hareket Vektörleri
        displacement = good_new - good_old
        dy = displacement[:, 1] # Dikey hareket (Yolu takip ediyoruz)
        
        # --- FILTRELEME (IQR / RANSAC Mantığı) ---
        if len(dy) > 0:
            q75, q25 = np.percentile(dy, [75 ,25])
            iqr = q75 - q25
            lower = q25 - (1.5 * iqr)
            upper = q75 + (1.5 * iqr)
            
            # Sadece güvenilir vektörleri al
            clean_dy = dy[(dy >= lower) & (dy <= upper)]
            
            if len(clean_dy) > 0:
                avg_dy = np.mean(clean_dy)
            else:
                avg_dy = 0
        else:
            avg_dy = 0
            
        # Hız Hesabı (Magnitude)
        # Genelde yol aşağı (pozitif) akar, biz ileri gideriz. Mutlak değer alıp çarpıyoruz.
        v_current = abs(avg_dy) * PIXEL_TO_METRIC
        est_speed.append(v_current)

        # --- GÖRSELLEŞTİRME ---
        # 1. ROI Alanını Çiz (Sarı)
        cv.polylines(vis_frame, [roi_pts], True, (0, 255, 255), 2)
        
        # 2. Vektörleri Çiz
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            # Outlier ise Kırmızı, Temiz ise Yeşil
            curr_dy = new[1] - old[1]
            if lower <= curr_dy <= upper:
                color = (0, 255, 0) # Yeşil (İyi)
            else:
                color = (0, 0, 255) # Kırmızı (Gürültü)
                
            vis_frame = cv.line(vis_frame, (int(a), int(b)), (int(c), int(d)), color, 2)
            vis_frame = cv.circle(vis_frame, (int(a), int(b)), 3, color, -1)
            
        cv.putText(vis_frame, f"Tahmini Hiz: {v_current:.2f} m/s", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        est_speed.append(0)

    cv.imshow('TUSAS Vision Odometry (ROI Fixed)', vis_frame)
    
    k = cv.waitKey(1) & 0xff
    if k == 27: break

cap.release()
cv.destroyAllWindows()

# ==========================================
# 4. SONUÇ VE KALİBRASYON ANALİZİ
# ==========================================
est_speed = np.array(est_speed)
est_time = np.arange(len(est_speed)) / VIDEO_FPS

# Zaman eşitleme (Kırpma)
min_t = min(gt_time[-1], est_time[-1])
mask_gt = gt_time <= min_t
mask_est = est_time <= min_t

gt_t_crop = gt_time[mask_gt]
gt_v_crop = gt_speed[mask_gt]
est_t_crop = est_time[mask_est]
est_v_crop = est_speed[mask_est]

# Interpolasyon (Zamanları çakıştır)
est_v_interp = np.interp(gt_t_crop, est_t_crop, est_v_crop)

# SCALE FACTOR HESABI
if np.mean(est_v_interp) > 0.1:
    scale_factor = np.mean(gt_v_crop) / np.mean(est_v_interp)
else:
    scale_factor = 1.0

# Kalibre edilmiş hız
est_v_calibrated = est_v_interp * scale_factor
rmse = np.sqrt(np.mean((gt_v_crop - est_v_calibrated)**2))

print("\n" + "="*40)
print(f"ANALİZ SONUCU (ROI DÜZELTMESİ SONRASI)")
print("="*40)
print(f"Hesaplanan Scale Factor: {scale_factor:.4f}")
print(f"Kalibre Edilmiş Hata (RMSE): {rmse:.2f} m/s")
print("="*40)

# GRAFİK
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(gt_t_crop, gt_v_crop, 'r-', label='Ground Truth')
plt.plot(est_t_crop, est_v_crop, 'b--', alpha=0.6, label='Senin Kodun (Ham)')
plt.title('Ham Veri (Scale Edilmemiş)')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(gt_t_crop, gt_v_crop, 'r-', linewidth=2, label='Ground Truth')
plt.plot(gt_t_crop, est_v_calibrated, 'g-', linewidth=1.5, label='Senin Kodun (Kalibre)')
plt.title(f'KALİBRE EDİLMİŞ SONUÇ (RMSE: {rmse:.2f})')
plt.xlabel('Zaman (sn)')
plt.ylabel('Hız (m/s)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
