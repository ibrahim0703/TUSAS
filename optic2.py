import numpy as np
import cv2 as cv

# --- AYARLAR ---
video_path = 'drone_video.mp4' # İndirdiğin videonun adı
# Drone kamerasının yeri ne kadar hızlı taradığını anlamak için katsayı
# Bunu videoya göre deneme-yanılma ile bulacağız (Kalibrasyon)
SCALE_FACTOR = 1.0 

def select_grid_points(frame, step=20):
    """
    Ekrana eşit aralıklarla (Grid) nokta döşer.
    Shi-Tomasi yerine bunu kullanıyoruz çünkü zemin her yerde aynıdır.
    """
    h, w = frame.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    return np.array(list(zip(x, y)), dtype=np.float32).reshape(-1, 1, 2)

cap = cv.VideoCapture(video_path)

# Parametreler (Daha hassas)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
if not ret: exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# İlk noktaları GRID olarak al
p0 = select_grid_points(old_gray, step=30) # 30 pikselde bir nokta koy

print("Drone Vision Başlatıldı...")

while True:
    ret, frame = cap.read()
    if not ret: break
    vis_frame = frame.copy()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 1. NOKTA KONTROLÜ (Sürekli Grid Yenileme)
    # Drone hareket ettikçe noktalar ekrandan çıkar, yenilerini ekle.
    if p0 is None or len(p0) < 50:
        p0 = select_grid_points(frame_gray, step=30)
        old_gray = frame_gray.copy()
        continue

    # 2. OPTICAL FLOW
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # Hareket Vektörleri
        displacement = good_new - good_old
        dx = displacement[:, 0]
        dy = displacement[:, 1]
        
        # 3. HIZ HESABI (Drone Mantığı)
        # Tüm ekranın ortalama hareketini al (Global Motion)
        # RANSAC yerine basit ortalama veya medyan yeterlidir çünkü 
        # aşağı bakarken tüm ekran aynı yöne akar (araba gibi değil).
        vx_pixel = np.median(dx)
        vy_pixel = np.median(dy)
        
        # Gürültü Filtresi (Titreşim)
        if abs(vx_pixel) < 0.5: vx_pixel = 0
        if abs(vy_pixel) < 0.5: vy_pixel = 0
        
        # Piksel Hızını Göster
        cv.putText(vis_frame, f"Vx: {vx_pixel:.2f} px", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(vis_frame, f"Vy: {vy_pixel:.2f} px", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 4. GÖRSELLEŞTİRME (Akış Yönü)
        # Ekranın ortasından hareket yönüne bir ok çiz
        h, w = frame.shape[:2]
        center = (w//2, h//2)
        # Eksi ile çarpıyoruz çünkü zemin sola kayıyorsa drone sağa gidiyordur.
        end_point = (int(center[0] - vx_pixel*10), int(center[1] - vy_pixel*10))
        
        cv.arrowedLine(vis_frame, center, end_point, (0, 0, 255), 5)
        
        # Grid noktalarını çiz
        for new in good_new:
            a, b = new.ravel()
            cv.circle(vis_frame, (int(a), int(b)), 2, (0, 255, 0), -1)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        # Takip koparsa resetle
        p0 = select_grid_points(frame_gray, step=30)
        old_gray = frame_gray.copy()

    cv.imshow('Drone Optical Flow', vis_frame)
    if cv.waitKey(30) & 0xff == 27: break

cap.release()
cv.destroyAllWindows()
