import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import traceback

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
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # --- GÜNCELLENMİŞ GENİŞ AÇI (WIDE ANGLE) ---
    # Yanları daha çok açtık ki kaldırımları görsün.
    # Üstü biraz daha yukarı aldık ki ufku görsün.
    pts = np.array([
        [int(w * 0.05), int(h * 0.95)],  # Sol Alt (En köşe)
        [int(w * 0.30), int(h * 0.35)],  # Sol Üst (Ufka yakın)
        [int(w * 0.70), int(h * 0.35)],  # Sağ Üst
        [int(w * 0.95), int(h * 0.95)]   # Sağ Alt (En köşe)
    ], dtype=np.int32)

    pts = pts.reshape((-1, 1, 2)) 
    cv.fillPoly(mask, [pts], 255)
    return mask, pts

# ==========================================
# 3. BAŞLATMA
# ==========================================
print("Sistem başlatılıyor (Hassas Mod)...")

cap = cv.VideoCapture(video_path)
VIDEO_FPS = cap.get(cv.CAP_PROP_FPS)
if VIDEO_FPS == 0: VIDEO_FPS = 30.0
DT = 1.0 / VIDEO_FPS
PIXEL_TO_METRIC = HEIGHT_Z / (FOCAL_LENGTH * DT)

# --- KRİTİK DEĞİŞİKLİK: Feature Params ---
# qualityLevel: 0.2 -> 0.01 (Çok daha düşük kontrastlı noktaları da kabul et)
# minDistance: 7 -> 5 (Noktalar birbirine daha yakın olabilir)
# maxCorners: 150 -> 500 (Daha fazla nokta ara)
feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=7)

lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

ret, old_frame = cap.read()
if not ret:
    print("HATA: Video açılamadı!")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
roi_mask, roi_pts = get_roi_mask(old_frame)

# İlk noktaları bulmaya çalış
p0 = cv.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)

est_speed = []
frame_count = 0

print(f"Video Analizi Başladı... Çıkmak için pencereye tıklayıp 'ESC'ye bas.")

# ==========================================
# 4. ANA DÖNGÜ
# ==========================================
try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vis_frame = frame.copy() 
        
        # SARI ROI ÇİZİMİ (Her zaman çiz ki nereye baktığını gör)
        cv.polylines(vis_frame, [roi_pts], True, (0, 255, 255), 2)

        # 2. Nokta Kontrolü ve Yenileme
        if p0 is None or len(p0) < 20: # Eşik değerini 50'den 20'ye düşürdük
            # Yeniden tara
            p0 = cv.goodFeaturesToTrack(frame_gray, mask=roi_mask, **feature_params)
            old_gray = frame_gray.copy()
            
            # Hız verisini koru
            current_v = est_speed[-1] if est_speed else 0
            est_speed.append(current_v)
            
            # Durum mesajı
            msg = "Nokta Bulunamadi!" if p0 is None else f"Nokta Yenilendi: {len(p0)}"
            color = (0, 0, 255) if p0 is None else (0, 255, 255)
            cv.putText(vis_frame, msg, (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv.imshow('TUSAS Vision Odometry', vis_frame)
            if cv.waitKey(1) & 0xff == 27: break
            continue 

        # 3. Optical Flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # Hareket vektörleri
            displacement = good_new - good_old
            dy = displacement[:, 1]
            
            # IQR Filtreleme (Daha güvenli)
            if len(dy) > 5: # En az 5 nokta varsa filtrele
                q75, q25 = np.percentile(dy, [75 ,25])
                iqr = q75 - q25
                lower = q25 - (1.5 * iqr)
                upper = q75 + (1.5 * iqr)
                clean_dy = dy[(dy >= lower) & (dy <= upper)]
                avg_dy = np.mean(clean_dy) if len(clean_dy) > 0 else 0
            elif len(dy) > 0: # Az nokta varsa direkt ortalama al
                avg_dy = np.mean(dy)
            else:
                avg_dy = 0
            
            v_current = abs(avg_dy) * PIXEL_TO_METRIC
            est_speed.append(v_current)

            # Görselleştirme
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv.line(vis_frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv.circle(vis_frame, (int(a), int(b)), 3, (0, 0, 255), -1)

            cv.putText(vis_frame, f"Hiz: {v_current:.2f} m/s", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(vis_frame, f"Takip: {len(good_new)} nokta", (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            est_speed.append(0)

        cv.imshow('TUSAS Vision Odometry', vis_frame)
        if cv.waitKey(1) & 0xff == 27: break

except Exception as e:
    print(f"\nHATA: {e}")
    print(traceback.format_exc())

finally:
    cap.release()
    cv.destroyAllWindows()
    print("Kapatıldı.")

# GRAFİK ÇİZİMİ (Kısa versiyon)
if len(est_speed) > 10:
    plt.plot(est_speed)
    plt.title("Hız Profili (m/s)")
    plt.grid()
    plt.show()
