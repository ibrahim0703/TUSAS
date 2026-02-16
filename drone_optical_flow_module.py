import numpy as np
import cv2 as cv

# =============================================================================
# TUSAŞ / SAVUNMA SANAYİİ STANDARDI: YAPILANDIRMA VE SABİTLER
# =============================================================================
# Sistem Parametreleri (GNC - Guidance Navigation Control)
CONFIG = {
    'VIDEO_SOURCE': r'C:/Users/A12540/Desktop/a/test.mp4',  # Test videosu
    'SCALE_FACTOR': 0.5,  # İşlem yükünü azaltmak için küçültme oranı
    'FOCAL_LENGTH_PX': 800.0,  # Kamera Kalibrasyon Matrisi (fx)
    'DT': 1.0 / 30.0,  # Döngü süresi (30 FPS varsayımı)

    # RANSAC Ayarları (Gürültü ve Hareketli Nesne Filtreleme)
    'RANSAC_THRESHOLD': 5.0,  # Piksel sapma eşiği (Outlier tespiti için)

    # Optik Akış Parametreleri (Lucas-Kanade)
    'LK_PARAMS': dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
    )
}


# =============================================================================
# MODÜL 1: SENSÖR ARAYÜZÜ (HARDWARE INTERFACE LAYER)
# =============================================================================
class DroneSensorInterface:
    """
    Bu sınıf, gerçek donanımda (Pixhawk/FCU) çalışan sensör verilerini simüle eder.
    Mentör Notu: Gerçek uçuşta buradaki veriler ROS topic'lerinden veya
    MAVLink mesajlarından okunacaktır.
    """

    def __init__(self):
        # Başlangıç değerleri
        self.agl_height_meters = 30.0  # Lidar'dan gelen yükseklik (Above Ground Level)
        self.gyro_rates_rad_s = np.array([0.0, 0.0, 0.0])  # IMU Gyro verisi (p, q, r)

    def get_telemetry_packet(self):
        """
        Uçuş kontrolcüsünden gelen anlık veri paketi.
        """
        # --- SİMÜLASYON ---
        # Gerçekte burası: return sub.recv_match(type='ATTITUDE')

        # 1. Lidar Verisi (Hafif gürültü eklenmiş simülasyon)
        noise = np.random.normal(0, 0.05)
        current_agl = max(2.0, self.agl_height_meters + noise)  # 2m altına düşmesin

        # 2. IMU Gyro Verisi (Dönüş hızları)
        # Optik akıştaki "Dönüş Kaynaklı Hata"yı (Rotation Compensation)
        # düzeltmek için bu veriye ihtiyacımız var.
        gyro_data = np.array([0.01, -0.01, 0.0])  # Hafif titreme simülasyonu

        return {
            'lidar_agl': current_agl,  # m (Metre)
            'imu_gyro': gyro_data,  # rad/s (Radyan/Saniye)
            'valid': True  # Veri geçerlilik bayrağı
        }


# =============================================================================
# MODÜL 2: YARDIMCI FONKSİYONLAR (UTILS)
# =============================================================================
def generate_grid_features(frame, step=30):
    """
    Shi-Tomasi yerine Grid tabanlı özellik noktası seçimi.
    Texture-rich (orman, asfalt) alanlar için homojen dağılım sağlar.
    """
    h, w = frame.shape[:2]
    margin = int(h * 0.1)  # Kenarlarda %10 ölü bölge bırak (Lens bozulması için)
    y, x = np.mgrid[margin:h - margin:step, margin:w - margin:step].reshape(2, -1).astype(int)
    return np.array(list(zip(x, y)), dtype=np.float32).reshape(-1, 1, 2)


def compensated_flow_to_metric_velocity(flow_px, agl, focal_len, dt, gyro):
    """
    Temel Optik Akış Denklemi: V_gerçek = (V_piksel * H) / f

    TODO (Gelecek Aşama): IMU verisi ile rotasyon kompanzasyonu:
    v_trans = v_obs - v_rot
    Burada sadece basit translational model uygulanmıştır.
    """
    # Piksel hızı (px/s)
    v_px_per_sec = flow_px / dt

    # Metrik hız (m/s) - Pinhole Model
    v_metric = (v_px_per_sec * agl) / focal_len

    return v_metric


# =============================================================================
# ANA ÇALIŞMA DÖNGÜSÜ (MAIN LOOP)
# =============================================================================
def main():
    # 1. Sistem Başlatma
    cap = cv.VideoCapture(CONFIG['VIDEO_SOURCE'])
    if not cap.isOpened():
        print("!!! KRİTİK HATA !!!")
        print(f"Video dosyası açılamadı: {CONFIG['VIDEO_SOURCE']}")
        print("Lütfen dosya ismini kontrol et veya tam yol (C:/...) kullan.")
        return

    sensors = DroneSensorInterface()

    ret, old_frame = cap.read()
    if not ret: 
        print("!!! HATA !!! Video açıldı ama ilk kare okunamadı.")
        return

    # Ön İşleme
    old_frame = cv.resize(old_frame, None, fx=CONFIG['SCALE_FACTOR'], fy=CONFIG['SCALE_FACTOR'])
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = generate_grid_features(old_gray, step=40)

    print("[INFO] Sistem Baslatildi. Yon Hatasi Duzeltildi.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        telemetry = sensors.get_telemetry_packet()
        current_agl = telemetry['lidar_agl']
        current_gyro = telemetry['imu_gyro']

        frame = cv.resize(frame, None, fx=CONFIG['SCALE_FACTOR'], fy=CONFIG['SCALE_FACTOR'])
        vis_frame = frame.copy()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if p0 is None or len(p0) < 50:
            p0 = generate_grid_features(frame_gray, step=40)
            old_gray = frame_gray.copy()
            continue

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **CONFIG['LK_PARAMS'])

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # RANSAC
            M, mask = cv.findHomography(good_old, good_new, cv.RANSAC, CONFIG['RANSAC_THRESHOLD'])

            if M is None: continue

            matchesMask = mask.ravel().tolist()

            inlier_vectors_x = []
            inlier_vectors_y = []

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                if matchesMask[i]:
                    dx = new[0] - old[0]
                    dy = new[1] - old[1]
                    inlier_vectors_x.append(dx)
                    inlier_vectors_y.append(dy)

                    # Yeşil Çizgiler (Zeminin akışını gösterir - AŞAĞI DOĞRU olmalı)
                    cv.line(vis_frame, (int(new[0]), int(new[1])), (int(old[0]), int(old[1])), (0, 255, 0), 1)
                else:
                    cv.circle(vis_frame, (int(new[0]), int(new[1])), 2, (0, 0, 255), -1)

            if len(inlier_vectors_x) > 0:
                mean_flow_x = np.mean(inlier_vectors_x)
                mean_flow_y = np.mean(inlier_vectors_y)

                vx_metric = compensated_flow_to_metric_velocity(mean_flow_x, current_agl, CONFIG['FOCAL_LENGTH_PX'],
                                                                CONFIG['DT'], current_gyro)
                vy_metric = compensated_flow_to_metric_velocity(mean_flow_y, current_agl, CONFIG['FOCAL_LENGTH_PX'],
                                                                CONFIG['DT'], current_gyro)

                # --- DÜZELTME BURADA ---
                # Zemin AŞAĞI (+y) akıyorsa, Drone İLERİ gidiyordur.
                # Hızın pozitif görünmesi için işaretleri ayarlıyoruz.
                v_forward = vy_metric  # Pozitif olması lazım (Drone frame +X)
                v_right = vx_metric  # Sağa kayma

                # --- HUD / GÖRSELLEŞTİRME ---
                cv.putText(vis_frame, f"AGL: {current_agl:.1f} m", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 255, 255), 2)

                # Renk Ayarı: İleri gidiyorsa YEŞİL, Geri gidiyorsa KIRMIZI yazı
                color_speed = (0, 255, 0) if v_forward > 0 else (0, 0, 255)
                cv.putText(vis_frame, f"HIZ (Ileri): {v_forward:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                           color_speed, 2)

                # --- OK YÖNÜ DÜZELTMESİ ---
                # Eskiden: (center + flow) -> Zemin nereye akıyorsa orayı gösteriyordu (Aşağı).
                # Şimdi: (center - flow) -> Drone nereye gidiyorsa orayı gösterecek (Yukarı).
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)

                # Oku ters çevir (-mean_flow) ve biraz büyüt (*20)
                end_pt = (int(center[0] - mean_flow_x * 20), int(center[1] - mean_flow_y * 20))

                cv.arrowedLine(vis_frame, center, end_pt, (0, 255, 255), 4, tipLength=0.3)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        else:
            p0 = generate_grid_features(frame_gray, step=40)
            old_gray = frame_gray.copy()

        cv.imshow('TUSAS Optical Flow Module (Level 1)', vis_frame)
        if cv.waitKey(int(CONFIG['DT'] * 1000)) & 0xff == 27: break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
