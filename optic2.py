import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import traceback # Hata takibi için

# ==========================================
# 1. AYARLAR
# ==========================================
video_path = 'araba_videosu.mp4'       
ground_truth_path = 'hiz_verisi.txt'   
GT_FREQ = 20.0                         

# Fiziksel Parametreler
FOCAL_LENGTH = 1000.0 
HEIGHT_Z = 1.5       

# ==========================================
# 2. YARDIMCI FONKSİYONLAR
# ==========================================
def get_roi_mask(frame):
    """
    Görüntünün sadece YOL kısmını alan maske.
    HATA DÜZELTİLDİ: Değişken tanımları ayrıldı.
    """
    # ÖNCE boyutu al (Syntax hatası buradaydı, düzeltildi)
    h, w = frame.shape[:2]
    
    # SONRA maskeyi oluştur
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Yamuk (Trapezoid) alanı
    # Köşeli parantezlere dikkat!
    pts = np.array([
        [int(w * 0.1), int(h * 0.90)],  # Sol Alt 
        [int(w * 0.35), int(h * 0.45)], # Sol Üst 
        [int(w * 0.65), int(h * 0.45)], # Sağ Üst
        [int(w * 0.9), int(h * 0.90)]   # Sağ Alt
    ], dtype=np.int32)

    pts = pts.reshape((-1, 1, 2)) 
    
    cv.fillPoly(mask, [pts], 255)
    return mask, pts

# ==========================================
# 3. BAŞLATMA
# ==========================================
print("Sistem başlatılıyor...")

# Veri Yükleme (Hata olsa da devam et)
gt_speed = np.zeros(100)
gt_time = np.arange(100) / GT_FREQ
try:
    temp_gt = np.loadtxt(ground_truth_path)
    gt_speed = temp_gt.reshape(-1)
    gt_time = np.arange(len(gt_speed)) / GT_FREQ
    print(f"-> Veri yüklendi: {len(gt_speed)} satır.")
except:
    print("-> Veri dosyası bulunamadı, sadece video modu çalışacak.")

cap = cv.VideoCapture(video_path)
VIDEO_FPS = cap.get(cv.CAP_PROP_FPS)
if VIDEO_FPS == 0: VIDEO_FPS = 30.0
DT = 1.0 / VIDEO_FPS
PIXEL_TO_METRIC = HEIGHT_Z / (FOCAL_LENGTH * DT)

# Parametreler
feature_params = dict(maxCorners=150, qualityLevel=0.2, minDistance=7, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

ret, old_frame = cap.read()
if not ret:
    print("HATA: Video açılamadı!")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
roi_mask, roi_pts = get_roi_mask(old_frame) # Maskeyi al
p0 = cv.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)

est_speed = []

print(f"Video Analizi Başladı... Çıkmak için pencereye tıklayıp 'ESC'ye bas.")

# ==========================================
# 4. ANA DÖNGÜ (DÜZELTİLMİŞ)
# ==========================================
try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vis_frame = frame.copy() # Ekrana basılacak temiz kare
        
        # 1. ROI Çizimi (Her karede çiz ki görelim)
        cv.polylines(vis_frame, [roi_pts], True, (0, 255, 255), 2)

        # 2. Nokta Kontrolü
        # Eğer nokta yoksa YENİLE ama 'continue' yapma, aşağı akış devam etsin
        if p0 is None or len(p0) < 50:
            p0 = cv.goodFeaturesToTrack(frame_gray, mask=roi_mask, **feature_params)
            old_gray = frame_gray.copy()
            # Hızı koru (veri yoksa 0)
            current_v = est_speed[-1] if est_speed else 0
            est_speed.append(current_v)
            
            # Ekrana durumu yaz
            cv.putText(vis_frame, "Nokta Araniyor...", (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # PENCEREYİ GÜNCELLE VE DEVAM ET (Donmayı önler)
            cv.imshow('TUSAS Vision Odometry', vis_frame)
            if cv.waitKey(1) & 0xff == 27: break
            continue 

        # 3. Optical Flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            displacement = good_new - good_old
            dy = displacement[:, 1]
            
            # IQR Filtreleme
            if len(dy) > 0:
                q75, q25 = np.percentile(dy, [75 ,25])
                iqr = q75 - q25
                lower = q25 - (1.5 * iqr)
                upper = q75 + (1.5 * iqr)
                clean_dy = dy[(dy >= lower) & (dy <= upper)]
                
                if len(clean_dy) > 0:
                    avg_dy = np.mean(clean_dy)
                else:
                    avg_dy = 0
            else:
                avg_dy = 0
            
            # Hız Hesabı
            v_current = abs(avg_dy) * PIXEL_TO_METRIC
            est_speed.append(v_current)

            # Çizimler
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # Outlier kontrolü (Renklendirme)
                delta_y = new[1] - old[1]
                if lower <= delta_y <= upper:
                    color = (0, 255, 0) # Yeşil (Güvenilir)
                else:
                    color = (0, 0, 255) # Kırmızı (Gürültü)
                
                cv.line(vis_frame, (int(a), int(b)), (int(c), int(d)), color, 2)
                cv.circle(vis_frame, (int(a), int(b)), 3, color, -1)

            cv.putText(vis_frame, f"Hiz: {v_current:.2f} m/s", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            est_speed.append(0)

        # 4. GÖSTERİM (Döngü her döndüğünde burası çalışmalı)
        cv.imshow('TUSAS Vision Odometry', vis_frame)
        
        # Bekleme süresi (Windows'un donmaması için kritik)
        if cv.waitKey(1) & 0xff == 27: # ESC
            break

except Exception as e:
    print("\n" + "!"*30)
    print("HATA OLUŞTU!")
    print(traceback.format_exc()) # Hatanın tam yerini gösterir
    print("!"*30)

finally:
    cap.release()
    cv.destroyAllWindows()
    print("Program kapatıldı.")

# ==========================================
# 5. GRAFİK (Eğer en az 10 veri varsa çiz)
# ==========================================
if len(est_speed) > 10:
    est_speed = np.array(est_speed)
    est_time = np.arange(len(est_speed)) / VIDEO_FPS
    
    # Kırpma ve Eşitleme
    min_t = min(gt_time[-1], est_time[-1])
    gt_crop = gt_speed[gt_time <= min_t]
    gt_t_crop = gt_time[gt_time <= min_t]
    
    est_crop = est_speed[est_time <= min_t]
    est_t_crop = est_time[est_time <= min_t]
    
    # İnterpolasyon
    est_interp = np.interp(gt_t_crop, est_t_crop, est_crop)
    
    # Scale Factor
    if np.mean(est_interp) > 0.1:
        scale = np.mean(gt_crop) / np.mean(est_interp)
    else:
        scale = 1.0
        
    est_calib = est_interp * scale
    rmse = np.sqrt(np.mean((gt_crop - est_calib)**2))
    
    print(f"\nSONUÇ: Scale Factor = {scale:.4f} | RMSE = {rmse:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_t_crop, gt_crop, 'r-', label='Gerçek Veri')
    plt.plot(gt_t_crop, est_calib, 'g-', label=f'Senin Kodun (RMSE: {rmse:.2f})')
    plt.title("Sonuç Karşılaştırması")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Yeterli veri toplanmadan çıkıldı.")
