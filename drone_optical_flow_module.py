import numpy as np
import cv2 as cv

# =============================================================================
# 1. KONFİGÜRASYON (AYARLAR)
# =============================================================================
CONFIG = {
    # --- VİDEO YOLU (Bunu kendi bilgisayarına göre değiştir) ---
    'VIDEO_SOURCE': r'C:/Users/A12540/Desktop/a/test.mp4', 
    
    'SCALE_FACTOR': 0.5,       # İşlem hızını artırmak için küçültme (0.5 = %50)
    'FOCAL_LENGTH_PX': 800.0,  # Kamera Odak Uzaklığı (Varsayılan)
    
    # --- ROI (GÖKYÜZÜ MASKELEME) ---
    # Eğer videoda ufuk çizgisi varsa, üst kısmı kesmek için burayı artır (0.0 ile 1.0 arası)
    # Örn: 0.40 yaparsan ekranın üst %40'ını görmezden gelir.
    'ROI_SKY_MASK_PERCENT': 0.40, 

    # --- RANSAC & OPTICAL FLOW ---
    'RANSAC_THRESHOLD': 5.0,   # Hata toleransı
    'LK_PARAMS': dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
    )
}

# =============================================================================
# 2. DRONE SENSÖR ARAYÜZÜ (SIMULATED / REAL)
# =============================================================================
class DroneSensorInterface:
    """
    Bu sınıf, Lidar ve IMU verilerini sağlar.
    Şu an Simülasyon modundadır. İleride CSV okuyacak şekilde güncellenecektir.
    """
    def __init__(self):
        # Varsayılan yükseklik (Video analizine göre değiştirebilirsin)
        self.agl_height_meters = 20.0 
        self.gyro_rates_rad_s = np.array([0.0, 0.0, 0.0]) # [Roll, Pitch, Yaw] hızı

    def get_telemetry_packet(self):
        """
        Her karede çağrılır. Anlık sensör verisini döndürür.
        """
        # --- SİMÜLASYON KISMI ---
        # Gerçek uçuşta burası sensörden okuyacak.
        
        # Hafif gürültü ekle (Gerçekçilik için)
        noise = np.random.normal(0, 0.05)
        current_agl = max(2.0, self.agl_height_meters + noise)

        # Dönüş verisi (Şimdilik 0 kabul ediyoruz)
        current_gyro = np.array([0.0, 0.0, 0.0]) 

        return {
            'lidar_agl': current_agl,  # Metre
            'imu_gyro': current_gyro,  # Rad/s
            'valid': True
        }

# =============================================================================
# 3. YARDIMCI MATEMATİK FONKSİYONLARI
# =============================================================================
def generate_grid_features(frame, step=30):
    """Ekrana eşit aralıklarla nokta döşer."""
    h, w = frame.shape[:2]
    margin = int(h * 0.1)
    y, x = np.mgrid[margin:h-margin:step, margin:w-margin:step].reshape(2, -1).astype(int)
    return np.array(list(zip(x, y)), dtype=np.float32).reshape(-1, 1, 2)

def compensated_flow_to_metric_velocity(flow_px, agl, focal_len, dt, gyro):
    """
    Piksel kaymasını (px) -> Gerçek hıza (m/s) çevirir.
    Level 2'de buraya Gyro düzeltmesi eklenecektir.
    """
    # 1. Piksel Hızı (px/s)
    v_px_per_sec = flow_px / dt

    # 2. Metrik Hız (m/s) - Pinhole Model
    # V = (v_px * H) / f
    v_metric = (v_px_per_sec * agl) / focal_len

    return v_metric

# =============================================================================
# 4. ANA PROGRAM (MAIN LOOP)
# =============================================================================
def main():
    print(f"[BAŞLATILIYOR] Video: {CONFIG['VIDEO_SOURCE']}")
    
    # Videoyu Aç
    cap = cv.VideoCapture(CONFIG['VIDEO_SOURCE'])
    if not cap.isOpened():
        print("!!! KRİTİK HATA !!! Video dosyası bulunamadı/açılamadı.")
        return

    # Otomatik FPS Algılama
    real_fps = cap.get(cv.CAP_PROP_FPS)
    if real_fps > 0 and not np.isnan(real_fps):
        dt = 1.0 / real_fps
        print(f"[BİLGİ] FPS: {real_fps:.2f} | DT: {dt:.4f} sn")
    else:
        dt = 1.0 / 30.0
        print("[UYARI] FPS okunamadı, 30 FPS varsayılıyor.")

    # Sensör Arayüzünü Başlat
    sensors = DroneSensorInterface()
    # İstersen yüksekliği buradan manuel düzeltebilirsin:
    sensors.agl_height_meters = 50.0 

    ret, old_frame = cap.read()
    if not ret: return

    # --- ROI (GÖKYÜZÜ KESME) AYARI ---
    h_raw, w_raw = old_frame.shape[:2]
    roi_start_y = int(h_raw * CONFIG['ROI_SKY_MASK_PERCENT']) # Üst kısmı at
    
    # İlk kareyi işle (Kes -> Küçült -> Gri Yap)
    old_frame_roi = old_frame[roi_start_y:h_raw, 0:w_raw]
    old_frame_resized = cv.resize(old_frame_roi, None, fx=CONFIG['SCALE_FACTOR'], fy=CONFIG['SCALE_FACTOR'])
    old_gray = cv.cvtColor(old_frame_resized, cv.COLOR_BGR2GRAY)
    
    # İlk noktaları oluştur
    p0 = generate_grid_features(old_gray, step=40)

    print("[BİLGİ] Analiz Döngüsü Başladı...")

    while True:
        ret, frame = cap.read()
        if not ret: 
            # Video bitince başa sar
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            p0 = None
            continue

        # Sensör Verisini Al
        telemetry = sensors.get_telemetry_packet()
        current_agl = telemetry['lidar_agl']
        current_gyro = telemetry['imu_gyro']

        # Görüntüyü Kes ve Hazırla
        frame_roi = frame[roi_start_y:h_raw, 0:w_raw] # Gökyüzünü at
        vis_frame = cv.resize(frame_roi, None, fx=CONFIG['SCALE_FACTOR'], fy=CONFIG['SCALE_FACTOR'])
        frame_gray = cv.cvtColor(vis_frame, cv.COLOR_BGR2GRAY)

        # Nokta Takibi (Optical Flow)
        if p0 is None or len(p0) < 50:
            p0 = generate_grid_features(frame_gray, step=40)
            old_gray = frame_gray.copy()
        else:
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **CONFIG['LK_PARAMS'])

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                # Yeterli nokta var mı?
                if len(good_new) < 4:
                    p0 = generate_grid_features(frame_gray, step=40)
                    old_gray = frame_gray.copy()
                    continue

                # RANSAC (Outlier Temizliği)
                M, mask = cv.findHomography(good_old, good_new, cv.RANSAC, CONFIG['RANSAC_THRESHOLD'])

                if M is not None:
                    matchesMask = mask.ravel().tolist()
                    inlier_vectors_x = []
                    inlier_vectors_y = []

                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        if matchesMask[i]: # Eğer nokta sağlamsa (Inlier)
                            dx = new[0] - old[0]
                            dy = new[1] - old[1]
                            inlier_vectors_x.append(dx)
                            inlier_vectors_y.append(dy)
                            # Yeşil çizgi çiz
                            cv.line(vis_frame, (int(new[0]), int(new[1])), (int(old[0]), int(old[1])), (0, 255, 0), 1)

                    if len(inlier_vectors_x) > 0:
                        # Ortalama Piksel Hızı
                        mean_flow_x = np.mean(inlier_vectors_x)
                        mean_flow_y = np.mean(inlier_vectors_y)

                        # m/s Dönüşümü
                        vx_metric = compensated_flow_to_metric_velocity(mean_flow_x, current_agl, CONFIG['FOCAL_LENGTH_PX'], dt, current_gyro)
                        vy_metric = compensated_flow_to_metric_velocity(mean_flow_y, current_agl, CONFIG['FOCAL_LENGTH_PX'], dt, current_gyro)

                        # Koordinat Dönüşümü (İleri Kamera)
                        # Akış Aşağı (+Y) ise Drone İleri (+X) gidiyordur.
                        v_forward = vy_metric
                        v_right = vx_metric

                        # --- GÖRSELLEŞTİRME ---
                        color_speed = (0, 255, 0) if v_forward > 0 else (0, 0, 255)
                        cv.putText(vis_frame, f"HIZ: {v_forward:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, color_speed, 2)
                        
                        # Ok Yönü (Akışın tersi)
                        h, w = vis_frame.shape[:2]
                        cx, cy = w // 2, h // 2
                        end_pt = (int(cx - mean_flow_x * 10), int(cy - mean_flow_y * 10))
                        cv.arrowedLine(vis_frame, (cx, cy), end_pt, (0, 255, 255), 4, tipLength=0.3)

                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = generate_grid_features(frame_gray, step=40)
                old_gray = frame_gray.copy()

        # Bilgi Ekranı
        cv.putText(vis_frame, f"AGL: {current_agl:.1f}m", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv.imshow('TUSAS Optical Flow Module (Final)', vis_frame)
        
        # FPS Kontrolü
        delay = int(1000 * dt)
        if cv.waitKey(delay) & 0xff == 27: # ESC ile çık
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
