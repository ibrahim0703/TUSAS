def remove_rotation_from_flow(flow_x, flow_y, gyro_x, gyro_y, focal_length, dt):
    """
    flow_x, flow_y : Optik akıştan gelen ham piksel kayması (piksel)
    gyro_x, gyro_y : IMU'dan gelen açısal hız (radyan/saniye)
    focal_length   : Kamera odak uzaklığı (piksel cinsinden)
    dt             : Geçen süre (saniye)
    """
    
    # 1. IMU verisine göre teorik piksel kaymasını hesapla
    # Formül (Basitleştirilmiş): Piksel = Açı * Odak_Uzaklığı
    # Açı = Açısal_Hız * Süre
    
    angle_x = gyro_x * dt  # Radyan cinsinden ne kadar döndük?
    angle_y = gyro_y * dt
    
    # Dönüş kaynaklı piksel kayması (Expected Flow due to Rotation)
    # İşaretler kamera montajına göre değişir (+/-), deneyerek bulunur.
    # Genellikle: Kafa aşağı dönerse (+Pitch), görüntü yukarı kayar (-Y).
    rot_flow_x = -angle_y * focal_length 
    rot_flow_y = angle_x * focal_length  
    
    # 2. Toplam akıştan bu "sahte" kısmı çıkar
    true_flow_x = flow_x - rot_flow_x
    true_flow_y = flow_y - rot_flow_y
    
    return true_flow_x, true_flow_y

import pandas as pd

class DroneSensorInterface:
    def __init__(self, csv_path):
        # CSV dosyasını yükle
        # EuRoC formatı: timestamp, w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, ...
        self.data = pd.read_csv(csv_path)
        self.current_idx = 0
        
    def get_synced_imu(self, video_timestamp):
        """
        Videonun o anki saniyesine en yakın IMU verisini bulup getirir.
        """
        # (Burada timestamp eşleştirme kodu olacak)
        # Şimdilik basitçe sıradaki veriyi döndürüyoruz diyelim:
        row = self.data.iloc[self.current_idx]
        self.current_idx += 1
        
        gyro_x = row['w_RS_S_x']
        gyro_y = row['w_RS_S_y']
        
        return np.array([gyro_x, gyro_y, 0.0])
