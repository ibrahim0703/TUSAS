import cv2 as cv
import os

# --- BURAYA VİDEONUN TAM YOLUNU YAPIŞTIR (Tırnak içine) ---
# ÖRNEK: r"C:\Users\Ogrenci\Desktop\Proje\video.mp4"
VIDEO_PATH = r"BURAYA_DOSYA_YOLUNU_YAPISTIR" 

print(f"--- TEŞHİS BAŞLIYOR ---")
print(f"OpenCV Versiyonu: {cv.__version__}")

# 1. Dosya Yolu Kontrolü
if os.path.exists(VIDEO_PATH):
    print(f"[OK] Dosya bulundu: {VIDEO_PATH}")
else:
    print(f"[HATA] Dosya bulunamadı! Yol yanlış: {VIDEO_PATH}")
    exit()

# 2. Video Açma Kontrolü
cap = cv.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[HATA] Dosya var ama OpenCV açamıyor!")
    print("       Sebepler: 1. Codec eksik (mp4 okuyamıyor).")
    print("       Sebepler: 2. ffmpeg dll dosyaları eksik.")
    print("       Çözüm: Videoyu .avi formatına çevirip dene.")
    exit()
else:
    print(f"[OK] VideoCapture nesnesi oluşturuldu. Backend: {cap.getBackendName()}")

# 3. Kare Okuma Kontrolü
ret, frame = cap.read()

if not ret:
    print("[HATA] Video açıldı ama ilk kare OKUNAMADI (ret=False).")
    print("       Dosya bozuk olabilir veya format desteklenmiyor.")
    exit()
else:
    print(f"[OK] İlk kare okundu. Boyut: {frame.shape}")

print("[BAŞARILI] Pencere açılmaya çalışılıyor...")

while True:
    cv.imshow('Test Penceresi', frame)
    
    ret, frame = cap.read()
    if not ret: break
    
    if cv.waitKey(30) & 0xff == 27: break

cap.release()
cv.destroyAllWindows()
