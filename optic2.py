import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ==========================================
# AYARLAR
# ==========================================
video_path = 'araba_videosu.mp4'
MAGIC_MULTIPLIER = 30.0 

# --- YENİ AYAR: TİTREŞİM FİLTRESİ ---
# Eğer bir nokta 2 pikselden az hareket ettiyse onu "Gürültü" sayıp atacağız.
MIN_PIXEL_MOVE = 2.0 

# ==========================================
# FONKSİYONLAR
# ==========================================
def get_roi_mask(frame):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # --- DÜZELTME: KAPUTU TAMAMEN AT ---
    # Alt sınırı 0.95'ten 0.80'e çektim. Kaput artık kadraja giremez.
    pts = np.array([
        [int(w * 0.10), int(h * 0.80)],  # Sol Alt (Daha yukarıda)
        [int(w * 0.35), int(h * 0.45)],  # Sol Üst
        [int(w * 0.65), int(h * 0.45)],  # Sağ Üst
        [int(w * 0.90), int(h * 0.80)]   # Sağ Alt (Daha yukarıda)
    ], dtype=np.int32).reshape((-1, 1, 2))
    
    cv.fillPoly(mask, [pts], 255)
    return mask, pts

# ==========================================
# ANA KOD
# ==========================================
cap = cv.VideoCapture(video_path)
feature_params = dict(maxCorners=300, qualityLevel=0.05, minDistance=5, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

ret, old_frame = cap.read()
if not ret: exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
roi_mask, roi_pts = get_roi_mask(old_frame)
p0 = cv.goodFeaturesToTrack(old_gray, mask=roi_mask, **feature_params)

est_speed = []

print("Filtreli Analiz Başladı...")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        vis_frame = frame.copy()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Sarı ROI'yi Çiz (Kontrol et, kaputun üstünde mi?)
        cv.polylines(vis_frame, [roi_pts], True, (0, 255, 255), 2)

        if p0 is None or len(p0) < 10:
            p0 = cv.goodFeaturesToTrack(frame_gray, mask=roi_mask, **feature_params)
            old_gray = frame_gray.copy()
            est_speed.append(0)
            cv.imshow('Filtered Vision', vis_frame)
            if cv.waitKey(1) & 0xff == 27: break
            continue

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # --- KRİTİK FİLTRELEME ADIMI ---
            clean_vectors = []
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                dx = new[0] - old[0]
                dy = new[1] - old[1]
                
                # 1. TİTREŞİM KONTROLÜ (Magnitude Check)
                magnitude = np.sqrt(dx**2 + dy**2)
                if magnitude < MIN_PIXEL_MOVE:
                    # Bu nokta titreşimdir, çizme ve hesaba katma
                    continue 

                # 2. YÖN KONTROLÜ (Direction Check)
                # Yol aşağı (pozitif y) akar. Yukarı gidenleri (negatif y) at.
                # Yanlara aşırı gidenleri (arabalar) at.
                if dy < 0.5: continue # Geriye gitme veya durma
                if abs(dx) > abs(dy): continue # Yanal hareket dikeyden fazlaysa (Yanından geçen araba)

                # Eğer buraya geldiyse bu SAĞLAM bir vektördür
                clean_vectors.append(dy)
                
                # Görselleştir (Sadece sağlamları yeşil çiz)
                a, b = new.ravel()
                c, d = old.ravel()
                cv.arrowedLine(vis_frame, (int(c), int(d)), (int(c), int(d+dy*5)), (0, 255, 0), 2)

            # --- HIZ HESABI ---
            if len(clean_vectors) > 0:
                avg_dy = np.mean(clean_vectors)
                v_current = avg_dy * MAGIC_MULTIPLIER
            else:
                v_current = 0.0 # Hiç sağlam vektör yoksa DURUYORUZ

            est_speed.append(v_current)

            cv.putText(vis_frame, f"Hiz: {v_current:.2f}", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv.putText(vis_frame, f"Aktif Vektor: {len(clean_vectors)}", (30, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            est_speed.append(0)

        cv.imshow('Filtered Vision', vis_frame)
        if cv.waitKey(1) & 0xff == 27: break

except Exception as e:
    print(e)
finally:
    cap.release()
    cv.destroyAllWindows()

# Grafik
plt.plot(est_speed)
plt.title("Filtrelenmiş Hız")
plt.show()
