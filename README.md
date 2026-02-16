def main():
    print(f"[DEBUG] Video yolu aranıyor: {CONFIG['VIDEO_SOURCE']}")
    
    # 1. Sistem Başlatma
    cap = cv.VideoCapture(CONFIG['VIDEO_SOURCE'])
    
    # VİDEO KONTROLÜ (Bunu ekle)
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

    # ... kodun geri kalanı aynı ...
