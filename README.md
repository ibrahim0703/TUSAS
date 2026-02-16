import numpy as np
import cv2 as cv
import os

# --- AYARLAR ---
# DİKKAT: Buraya videonun TAM YOLUNU yaz. (Örn: C:/Users/.../video.mp4)
VIDEO_PATH = '282749_medium.mp4' 
# EĞER HÂLÂ ÇALIŞMAZSA: Burayı 0 yapıp kameranı açmayı dene.
# VIDEO_PATH = 0 

def main():
    print("------------------------------------------------")
    print("[1] Program Başlıyor...")
    
    # Dosya Kontrolü
    if VIDEO_PATH != 0 and not os.path.exists(VIDEO_PATH):
        print(f"!!! HATA: Video dosyası bu konumda YOK: {VIDEO_PATH}")
        print("    -> Lütfen dosyanın tam yolunu (Full Path) yaz.")
        return

    print(f"[2] Video Yolu Doğru: {VIDEO_PATH}")
    
    cap = cv.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("!!! HATA: VideoCapture açılamadı. Codec sorunu olabilir.")
        return
    print("[3] VideoCapture nesnesi oluşturuldu.")

    ret, old_frame = cap.read()
    if not ret:
        print("!!! HATA: Video açıldı ama İLK KARE okunamadı!")
        print("    -> Dosya bozuk olabilir veya OpenCV bu formatı okuyamıyor.")
        return
    print("[4] İlk kare başarıyla okundu.")

    # Boyutlandırma
    try:
        old_frame = cv.resize(old_frame, None, fx=0.5, fy=0.5)
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        print("[5] İlk kare işlendi (Resize/Gray).")
    except Exception as e:
        print(f"!!! HATA: Görüntü işleme hatası: {e}")
        return

    print("[6] Döngüye giriliyor...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[BİLGİ] Video bitti veya kare okunamadı. Toplam Kare: {frame_count}")
            break
        
        frame_count += 1
        
        # Basit bir işlem yapalım (Karmaşıklığı eledik)
        frame = cv.resize(frame, None, fx=0.5, fy=0.5)
        
        # Ekrana bir şey yazalım ki çalıştığını görelim
        cv.putText(frame, f"Kare: {frame_count}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Pencere Açma (Zorla)
        cv.namedWindow('Test Penceresi', cv.WINDOW_NORMAL)
        cv.imshow('Test Penceresi', frame)
        
        # Bekleme
        if cv.waitKey(30) & 0xff == 27:
            print("[BİLGİ] Kullanıcı ESC ile çıktı.")
            break

    print("[SON] Program bitti.")
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
