import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ==========================================
# 1. AYARLAR
# ==========================================
video_path = 'araba_videosu.mp4'
ground_truth_path = 'hiz_verisi.txt'
GT_FREQ = 20.0

# --- SİHİRLİ DOKUNUŞ ---
# Kodun bulduğu o "0.09" gibi komik rakamları gerçek hıza (10-20 m/s)
# dönüştürmek için bu sayıyı kullanacağız.
# Eğer hız hala düşükse bunu 50, 100 yap. Çok yüksekse 10 yap.
MAGIC_MULTIPLIER = 30.0 

# ==========================================
# 2. YARDIMCI FONKSİYONLAR
# ==========================================
def get_roi_mask(frame):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # --- RADİKAL DEĞİŞİKLİK: SADECE YERİ GÖR ---
    # Ufuk çizgisini tamamen atıyoruz.
    # Ekranın sadece alt yarısına ve kenarlarına bakıyoruz.
    pts = np.array([
        [int(w * 0.05), int(h * 0.95)],  # Sol Alt
        [int(w * 0.20), int(h * 0.60)],  # Sol Üst (Ufuktan uzak dur)
        [int(w * 0.80), int(h * 0.60)],  # Sağ Üst
        [int(w * 0.95), int(h * 0.95)]   # Sağ Alt
    ], dtype=np.int32)

    pts = pts.reshape((-1, 1, 2))
    cv.fillPoly(mask, [pts], 255)
    return mask, pts

# ==========================================
# 3. BAŞLATMA
# ==========================================
# Veri Yükleme
gt_speed = np.zeros(100)
try:
    gt_speed = np.loadtxt(ground_truth_path).reshape(-1)
except:
    print("GT verisi yok, sadece video.")

cap = cv.VideoCapture(video_path)
feature_params = dict(maxCorners=300, qualityLevel=0.05, minDistance=5, blockSize=7) # Hassas ayar
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

ret, old_frame = cap.read()
if not ret: exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
roi_mask, roi_pts = get_roi_mask(old_frame)
p0 = cv.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)

est_speed = []
print("Analiz Başladı... (Çıkış: ESC)")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        vis_frame = frame.copy()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Sarı ROI Çizimi
        cv.polylines(vis_frame, [roi_pts], True, (0, 255, 255), 2)

        if p0 is None or len(p0) < 10:
            p0 = cv.goodFeaturesToTrack(frame_gray, mask=roi_mask, **feature_params)
            old_gray = frame_gray.copy()
            est_speed.append(est_speed[-1] if est_speed else 0)
            cv.imshow('Boosted Vision', vis_frame)
            if cv.waitKey(1) & 0xff == 27: break
            continue

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # Vektör Hesaplama
            displacement = good_new - good_old
            dy = displacement[:, 1]
            
            # --- FİLTRELEME VE BOOST ---
            if len(dy) > 0:
                # Sadece aşağı (pozitif) akanları al (Yol bize doğru akmaz, biz gideriz)
                # Gürültüyü (yukarı hareket edenleri) at.
                valid_dy = dy[dy > 0.1] 
                
                if len(valid_dy) > 0:
                    avg_dy = np.mean(valid_dy) # IQR yerine direkt ortalama (daha agresif)
                else:
                    avg_dy = 0
            else:
                avg_dy = 0
            
            # --- MAGIC MULTIPLIER DEVREDE ---
            # avg_dy (piksel) * Magic (Katsayı)
            v_current = avg_dy * MAGIC_MULTIPLIER
            est_speed.append(v_current)

            # --- GÖRSELLEŞTİRME (Vektörleri Uzat) ---
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                # Hareket Yönünü Gösteren Oklar (Abartılı çizim x5)
                # Böylece minik hareketleri bile gözünle görürsün
                arrow_end = (int(c + (a-c)*5), int(d + (b-d)*5))
                
                cv.arrowedLine(vis_frame, (int(c), int(d)), arrow_end, (0, 255, 0), 2, tipLength=0.3)

            cv.putText(vis_frame, f"Hiz: {v_current:.2f}", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv.putText(vis_frame, f"Carpan: x{MAGIC_MULTIPLIER}", (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            est_speed.append(0)

        cv.imshow('Boosted Vision', vis_frame)
        if cv.waitKey(1) & 0xff == 27: break

except Exception as e:
    print(e)
finally:
    cap.release()
    cv.destroyAllWindows()

# Hızlı Grafik
if len(est_speed) > 10:
    plt.plot(est_speed, label='Boosted Speed')
    if len(gt_speed) > 0:
        # GT verisini basitçe ölçekle (görsel kıyas için)
        plt.plot(gt_speed[:len(est_speed)], label='GT', alpha=0.5)
    plt.legend()
    plt.show()
