import numpy as np
import cv2 as cv
import pandas as pd
import os

# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================
CONFIG = {
    # Video yolunu buraya yaz
    'VIDEO_SOURCE': r'C:/Users/A12540/Desktop/a/1.mp4',
    
    # Eğer elinde veri seti varsa CSV yolunu buraya yaz (Yoksa None kalsın, simüle eder)
    'IMU_CSV_PATH': None, # Örn: r'C:/Users/.../imu_data.csv'
    
    'SCALE_FACTOR': 0.8,       # İşlem hızını artırmak için küçültme (0.8 = %80) # video yavaş işleniyorsa düşürebilirsin
    'FOCAL_LENGTH_PX': 1000,  # Kamera Odak Uzaklığı (Varsayılan) # hesaplanan hız düşük kalıyorsa focal_lenght arttır deneme yanılmayla doğru piksel metre dönüşümünü bul.
    
    # --- ROI (GÖKYÜZÜ MASKELEME) ---
    # Eğer videoda ufuk çizgisi varsa, üst kısmı kesmek için burayı artır (0.0 ile 1.0 arası)
    # Örn: 0.40 yaparsan ekranın üst %40'ını görmezden gelir. Eğer kameram dikse bunu 0 yap direk görseli kesmeden işlem yaparız.
    # Hız 0 çıkarsa durgun noktaları kesmek için % yi arttır. 
    'ROI_SKY_MASK_PERCENT': 0.40, 

    # --- RANSAC_THRESHOLD ---
    'RANSAC_THRESHOLD': 5.0, # Kaç piksel saparsam noktayı referans olarak almıyım diye sorar.
                             # Video titrek ve gürültülüyse sayıyı arttırarak daha çok noktaya bakarsın / Eğer stabil bir videoysa daha az referans noktasına bakarsın hareketlileri daha kolay elersin
    # --- LUCAS KANDE PARAMETERS ---
    'LK_PARAMS': dict(winSize=(21, 21), # takip yavaş kalırsa drone daha hızlıysa win_size büyüt.
                    maxLevel=3,
                    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
}

# =============================================================================
# 2. HİBRİT SENSÖR ARAYÜZÜ ( CSV path girersen kullanır yoksa SIMULATION yapar.)
# =============================================================================
class DroneSensorInterface:
    """
    Bu sınıf hem veri yoksa simülasyon  yapar hem de CSV okuma yeteneğine sahip.
    """
    def __init__(self, csv_path=None):
        self.agl_height_meters = 20.0  # Drone'nın yerden yükseliğini ayarladığımız yer.
        self.use_csv = False
        self.data_idx = 0
        
        # Eğer CSV dosyası varsa yükle
        if csv_path and os.path.exists(csv_path):
            try:
                # EuRoC formatı varsayımı (timestamp, wx, wy, wz...)
                self.df = pd.read_csv(csv_path)
                self.use_csv = True
                print(f"[SENSÖR] CSV Yüklendi: {len(self.df)} satır.")
            except Exception as e:
                print(f"[SENSÖR HATASI] CSV okunamadı: {e}")
        
        if not self.use_csv:
            print("[SENSÖR] CSV yok, SİMÜLASYON MODU aktif (Yapay Gyro).")

    def get_telemetry_packet(self, current_frame_idx, fps):
        """
        Anlık Gyro ve Lidar verisini döndürür.
        """
        # --- A. CSV MODU (GERÇEK VERİ) ---
        if self.use_csv:
            # Videodaki kareye denk gelen satırı bul (Basit senkronizasyon)
            # Gerçekte timestamp eşleşmesi gerekir, burada satır satır gidiyoruz.
            if self.data_idx < len(self.df):
                row = self.df.iloc[self.data_idx]
                self.data_idx += 1
                # Sütun isimlerini veri setine göre ayarlamak gerekir!
                # Örnek: 'w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z'
                # Şimdilik rastgele bir değer dönüyor simülasyon mantığıyla:
                return {
                    'lidar_agl': 20.0,
                    'imu_gyro': np.array([0.0, 0.0, 0.0]), # Burayı CSV sütunlarına bağlayacağız
                    'valid': True
                }

        # --- B. SİMÜLASYON MODU (TEST İÇİN) ---
        else:
            # Drone sanki hafifçe sağa sola sallanıyormuş gibi yapay veri üret
            # Sinüs dalgası ile Pitch ve Roll hareketi simüle ediyoruz
            t = current_frame_idx / fps
            sim_pitch_rate = 0.05 * np.sin(t) # Radyan/saniye (Kafa sallama)
            sim_roll_rate  = 0.02 * np.cos(t) # Radyan/saniye (Yana yatma)
            
            # Gürültülü AGL
            noise = np.random.normal(0, 0.05)
            current_agl = max(2.0, self.agl_height_meters + noise)

            return {
                'lidar_agl': current_agl,
                'imu_gyro': np.array([sim_roll_rate, sim_pitch_rate, 0.0]), 
                'valid': True
            }

# =============================================================================
# 3. LEVEL 2: ROTATION COMPENSATION MATH
# =============================================================================
def generate_grid_features(frame, step=30):
    h, w = frame.shape[:2]
    margin = int(h * 0.1)
    y, x = np.mgrid[margin:h-margin:step, margin:w-margin:step].reshape(2, -1).astype(int)
    return np.array(list(zip(x, y)), dtype=np.float32).reshape(-1, 1, 2)

def compensated_flow_to_metric_velocity(flow_px_total, agl, focal_len, dt, gyro_rate):
    """
    *** LEVEL 2 KRİTİK FONKSİYON ***
    V_gerçek = V_gözlenen - V_dönüş
    """
    # 1. Gözlenen Toplam Hız (Piksel/Saniye)
    v_px_total_per_sec = flow_px_total / dt

    # 2. Dönüş Kaynaklı Sahte Hız (Rotation Flow)
    # Formül: V_rot = Omega (rad/s) * Focal (px)
    # Pitch hareketi Y ekseninde, Roll/Yaw hareketi X ekseninde kayma yapar.
    v_px_rotation = gyro_rate * focal_len
    
    # 3. Gerçek İlerleme Hızı (Translation Flow)
    v_px_translation = v_px_total_per_sec - v_px_rotation

    # 4. Metrik Dönüşüm (m/s)
    # V_metric = (V_px_trans * H) / f
    v_metric = (v_px_translation * agl) / focal_len

    return v_metric, v_px_total_per_sec, v_px_rotation

# =============================================================================
# 4. ANA PROGRAM
# =============================================================================
def main():
    print(f"[SİSTEM] Level 2 Başlatılıyor: Rotation Compensation Aktif")
    
    cap = cv.VideoCapture(CONFIG['VIDEO_SOURCE'])
    if not cap.isOpened():
        print("!!! HATA !!! Video açılamadı.")
        return

    # FPS Ayarı
    real_fps = cap.get(cv.CAP_PROP_FPS)
    dt = 1.0 / real_fps if real_fps > 0 else 1.0/30.0
    print(f"[BİLGİ] FPS: {real_fps:.2f} | DT: {dt:.4f}s")

    sensors = DroneSensorInterface(CONFIG['IMU_CSV_PATH'])
    sensors.agl_height_meters = 50.0 

    ret, old_frame = cap.read()
    if not ret: return

    # ROI Ayarı
    h_raw, w_raw = old_frame.shape[:2]
    roi_start_y = int(h_raw * CONFIG['ROI_SKY_MASK_PERCENT'])
    
    # İlk kare hazırlığı
    old_frame_roi = old_frame[roi_start_y:h_raw, 0:w_raw]
    old_frame_resized = cv.resize(old_frame_roi, None, fx=CONFIG['SCALE_FACTOR'], fy=CONFIG['SCALE_FACTOR'])
    old_gray = cv.cvtColor(old_frame_resized, cv.COLOR_BGR2GRAY)
    p0 = generate_grid_features(old_gray, step=40)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            p0 = None
            continue
        
        frame_idx += 1
        
        # --- ADIM 1: SENSÖR VERİSİ (GYRO) ---
        telemetry = sensors.get_telemetry_packet(frame_idx, real_fps)
        current_agl = telemetry['lidar_agl']
        # Gyro: [Roll_Rate, Pitch_Rate, Yaw_Rate]
        gyro_roll = telemetry['imu_gyro'][0]
        gyro_pitch = telemetry['imu_gyro'][1]

        # --- ADIM 2: OPTICAL FLOW ---
        frame_roi = frame[roi_start_y:h_raw, 0:w_raw]
        vis_frame = cv.resize(frame_roi, None, fx=CONFIG['SCALE_FACTOR'], fy=CONFIG['SCALE_FACTOR'])
        frame_gray = cv.cvtColor(vis_frame, cv.COLOR_BGR2GRAY)

        if p0 is None or len(p0) < 50:
            p0 = generate_grid_features(frame_gray, step=40)
            old_gray = frame_gray.copy()
        else:
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **CONFIG['LK_PARAMS'])

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                if len(good_new) < 4:
                    p0 = generate_grid_features(frame_gray, step=40)
                    old_gray = frame_gray.copy()
                    continue

                # RANSAC outliers eler sağlam olanları seçer ve tahmini yapar.
                M, mask = cv.findHomography(good_old, good_new, cv.RANSAC, CONFIG['RANSAC_THRESHOLD'])

                if M is not None:
                    matchesMask = mask.ravel().tolist()
                    inlier_dx = []
                    inlier_dy = []

                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        if matchesMask[i]:
                            inlier_dx.append(new[0] - old[0])
                            inlier_dy.append(new[1] - old[1])
                            # Yeşil çizgi (Ham akış)
                            cv.line(vis_frame, (int(new[0]), int(new[1])), (int(old[0]), int(old[1])), (0, 255, 0), 1)

                    if len(inlier_dx) > 0:
                        mean_dx = np.mean(inlier_dx)
                        mean_dy = np.mean(inlier_dy)

                        # --- ADIM 3: DÖNÜŞ DÜZELTMELİ HIZ HESABI ---
                        
                        # X Ekseni (Sağ/Sol) -> Roll dönüşünden etkilenir (Basit model)
                        # Veya Yaw'dan etkilenir. İleri bakan kamerada Roll görüntüyü döndürür.
                        # Biz burada basitlik için Pitch ve Yaw/Roll etkileşimini alıyoruz.
                        
                        # Y HIZI (İleri/Geri) -> Pitch (Kafa eğme) hareketinden etkilenir
                        # Eğer kafa aşağı inerse (+Pitch), görüntü yukarı kayar (-dy).
                        # Bu yüzden Pitch hızını telafi etmeliyiz.
                        vy_metric, vy_raw, vy_rot = compensated_flow_to_metric_velocity(
                            mean_dy, current_agl, CONFIG['FOCAL_LENGTH_PX'], dt, gyro_rate=gyro_pitch
                        )
                        
                        # X HIZI (Sağ/Sol) -> Yaw veya Roll
                        vx_metric, vx_raw, vx_rot = compensated_flow_to_metric_velocity(
                            mean_dx, current_agl, CONFIG['FOCAL_LENGTH_PX'], dt, gyro_rate=gyro_roll
                        )

                        v_forward = vy_metric
                        v_right = vx_metric

                        # --- GÖRSELLEŞTİRME ---
                        # 1. Hız Yazısı
                        cv.putText(vis_frame, f"HIZ: {v_forward:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # 2. Gyro Bilgisi (Ekrana basalım ki çalıştığını gör)
                        cv.putText(vis_frame, f"GYRO Pitch: {gyro_pitch:.3f} rad/s", (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
                        
                        # 3. Ok Yönü (Düzeltilmiş Hız)
                        h, w = vis_frame.shape[:2]
                        cx, cy = w // 2, h // 2
                        # Görselde akışın tersini (Drone'un gidiş yönünü) çiziyoruz
                        end_pt = (int(cx - mean_dx * 10), int(cy - mean_dy * 10))
                        cv.arrowedLine(vis_frame, (cx, cy), end_pt, (0, 255, 255), 4, tipLength=0.3)

                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = generate_grid_features(frame_gray, step=40)
                old_gray = frame_gray.copy()

        cv.putText(vis_frame, f"AGL: {current_agl:.1f}m (Simulated)", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv.imshow('TUSAS Level 2: Gyro Compensation', vis_frame)
        
        if cv.waitKey(int(1000 * dt)) & 0xff == 27: break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
