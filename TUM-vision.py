import cv2
import numpy as np

# NOT: Bu kodun çalışması için StereoRectifier ve StereoFlowTracker 
# sınıflarının bu dosyanın üst kısmında tanımlanmış olması gerekir.

def test_and_visualize_pipeline():
    # 1. Sınıfları Başlat (Matrisler RAM'e yükleniyor)
    print("[BİLGİ] Sistem başlatılıyor ve matrisler hesaplanıyor...")
    rectifier = StereoRectifier()
    tracker = StereoFlowTracker()

    # 2. Resim Yolları (Kendi klasörüne göre BURAYI GÜNCELLE!)
    # İkisi de aynı zaman damgasına (timestamp) sahip olmalı.
    left_path = 'SVO_Veri/cam0/data/1520530308199447626.png'
    right_path = 'SVO_Veri/cam1/data/1520530308199447626.png'

    img_left_raw = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img_right_raw = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    if img_left_raw is None or img_right_raw is None:
        print("[HATA] Resimler okunamadı! Dosya yollarını kontrol et.")
        return

    # 3. Aşama 1: Kusursuz Hizalama (Rectification)
    img_left_rect, img_right_rect = rectifier.process(img_left_raw, img_right_raw)

    # 4. Aşama 2: Stereo Eşleştirme (Optical Flow)
    good_left, good_right = tracker.get_stereo_matches(img_left_rect, img_right_rect)
    
    print(f"[SONUÇ] Epipolar testten geçen BAŞARILI eşleşme sayısı: {len(good_left)}")

    if len(good_left) == 0:
        print("[HATA] Hiç nokta eşleşmedi. Sistemi kontrol et.")
        return

    # =========================================================================
    # 5. GÖRSELLEŞTİRME (VİSUALİZATİON) MODÜLÜ
    # =========================================================================
    
    # Renkli çizim yapabilmek için siyah-beyaz resimleri BGR'a çeviriyoruz
    vis_left = cv2.cvtColor(img_left_rect, cv2.COLOR_GRAY2BGR)
    vis_right = cv2.cvtColor(img_right_rect, cv2.COLOR_GRAY2BGR)

    # İki resmi yan yana yapıştır (512x512 iki resim -> 1024x512 tek resim olur)
    vis_combined = np.hstack((vis_left, vis_right))
    w = vis_left.shape[1] # Sol resmin genişliği (512). Sağ resme çizgi çekerken X'e eklenecek.

    # Her bir eşleşen nokta çifti için çizim yap
    for i in range(len(good_left)):
        pt_l = (int(good_left[i][0]), int(good_left[i][1]))
        pt_r = (int(good_right[i][0]) + w, int(good_right[i][1])) # Sağdaki noktanın X'ini sağa kaydır

        # Sol noktayı KIRMIZI çiz
        cv2.circle(vis_combined, pt_l, 4, (0, 0, 255), -1)
        # Sağ noktayı YEŞİL çiz
        cv2.circle(vis_combined, pt_r, 4, (0, 255, 0), -1)
        
        # İki nokta arasına SARI çizgi çek
        cv2.line(vis_combined, pt_l, pt_r, (0, 255, 255), 1)

    # İlk noktanın Derinlik (Z) hesabını ekrana yazdırıp test edelim
    d = good_left[0][0] - good_right[0][0] # Disparity (X farkı)
    if d > 0:
        z = (190.978 * 0.1009) / d
        print(f"[MATEMATİK] İlk noktanın piksel kayması (Disparity): {d:.2f} piksel")
        print(f"[MATEMATİK] İlk noktanın kameraya uzaklığı (Derinlik): {z:.2f} metre")

    # Görüntüyü ekrana bas
    cv2.imshow("Stereo Eslesme ve Epipolar Cizgiler", vis_combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_and_visualize_pipeline()
