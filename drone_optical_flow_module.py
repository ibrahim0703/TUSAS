import numpy as np
import cv2 as cv
import pandas as pd
import os

# =============================================================================
# 1. KONFİGÜRASYON
# =============================================================================
CONFIG = {
    'CAM0_CSV': 'cam0/data.csv',
    'CAM0_DIR': 'cam0/data',
    'CAM1_CSV': 'cam1/data.csv',
    'CAM1_DIR': 'cam1/data',
}

# =============================================================================
# 2. STEREO DATA LOADER (Donanımsal Senkronizasyon)
# =============================================================================
class StereoDataLoader:
    def __init__(self):
        print("[SİSTEM] Stereo Kameralar (Cam0 ve Cam1) Yükleniyor...")
        
        # Sol ve Sağ kamera kayıt defterlerini oku
        df0 = pd.read_csv(CONFIG['CAM0_CSV'])
        df0.columns = ['timestamp', 'filename']
        
        df1 = pd.read_csv(CONFIG['CAM1_CSV'])
        df1.columns = ['timestamp', 'filename']
        
        # EuRoC'ta iki kamera donanımsal olarak aynı nanosaniyede tetiklenir.
        # Yine de en yakın zaman damgalarını eşleştirerek sağlamlaştırıyoruz.
        df0 = df0.sort_values('timestamp')
        df1 = df1.sort_values('timestamp')
        
        # merge_asof: Cam0'daki her fotoğrafa, Cam1'den zamansal olarak en yakın fotoğrafı bağlar
        self.stereo_df = pd.merge_asof(df0, df1, on='timestamp', direction='nearest', suffixes=('_left', '_right'))
        print(f"[BİLGİ] {len(self.stereo_df)} adet Stereo Görüntü Çifti başarıyla senkronize edildi.")
        
        self.current_idx = 0
        self.total_frames = len(self.stereo_df)

    def get_stereo_pair(self):
        if self.current_idx >= self.total_frames:
            return None, None
            
        row = self.stereo_df.iloc[self.current_idx]
        
        img_left_path = os.path.join(CONFIG['CAM0_DIR'], str(row['filename_left']))
        img_right_path = os.path.join(CONFIG['CAM1_DIR'], str(row['filename_right']))
        
        img_left = cv.imread(img_left_path, cv.IMREAD_GRAYSCALE)
        img_right = cv.imread(img_right_path, cv.IMREAD_GRAYSCALE)
        
        self.current_idx += 1
        return img_left, img_right

# =============================================================================
# 3. STEREO SGBM (Semi-Global Block Matching) MOTORU
# =============================================================================
def init_stereo_matcher():
    """
    Sol ve Sağ kameralar arasındaki yatay piksel kaymasını (Disparity) hesaplar.
    Bu parametreler işlemciyi çok yormayacak şekilde endüstri standardı olan SGBM için ayarlandı.
    """
    window_size = 5 # Eşleştirilecek piksel bloğunun boyutu (5x5). Ne kadar küçükse o kadar detaylı ama gürültülü olur.
    min_disp = 0    # Minimum kayma miktarı (Sonsuzdaki nesneler 0 kayar)
    num_disp = 16 * 4 # Maksimum arayacağı kayma miktarı (16'nın katı olmak zorundadır). Derinlik sınırını belirler.
    
    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 1 * window_size ** 2,    # P1 ve P2 derinlik pürüzsüzlüğünü (smoothness) kontrol eden ceza parametreleridir.
        P2=32 * 1 * window_size ** 2,
        disp12MaxDiff=1, # Sol ve Sağ eşleştirmesi arasındaki maksimum tolerans
        uniquenessRatio=10, # Yanlış eşleşmeleri filtreleme sertliği
        speckleWindowSize=100, # Gürültü (speckle) lekelerini silme penceresi
        speckleRange=32
    )
    return stereo

# =============================================================================
# 4. ANA DÖNGÜ
# =============================================================================
def main():
    loader = StereoDataLoader()
    stereo_matcher = init_stereo_matcher()

    while True:
        img_left, img_right = loader.get_stereo_pair()
        if img_left is None:
            print("[SİSTEM] Veri Seti Bitti.")
            break
            
        # 1. Disparity'yi Hesapla
        # SGBM algoritması her zaman 16 ile çarpılmış bir matris döndürür. (OpenCV'nin veri yapısı gereği)
        disparity_16S = stereo_matcher.compute(img_left, img_right)
        
        # 2. Gerçek Piksel Kaymasına Çevir (16'ya bölerek)
        disparity_float = disparity_16S.astype(np.float32) / 16.0
        
        # 3. İnsan Gözünün Görebilmesi İçin Görselleştirme (Normalize etme)
        # Disparity değerlerini 0 (Siyah) ile 255 (Beyaz) arasına sıkıştırıyoruz.
        # Açık renk (Beyaz) = Çok kaymış = KAMERAYA ÇOK YAKIN
        # Koyu renk (Siyah) = Az kaymış = KAMERADAN ÇOK UZAK
        disp_vis = cv.normalize(disparity_float, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        # Daha havalı bir endüstriyel görünüm için renk haritası (ColorMap) uygulayalım
        disp_color = cv.applyColorMap(disp_vis, cv.COLORMAP_JET)

        # Kameraları ve Derinliği yan yana göster
        cv.imshow('Sol Kamera (Cam0)', img_left)
        cv.imshow('Derinlik (Disparity Map)', disp_color)

        if cv.waitKey(30) & 0xff == 27: # ESC ile çıkış
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
