import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# --- 1. AYARLAR ---
video_path = 'araba_videosu.mp4'      # Videonun adı
ground_truth_path = 'hiz_verisi.txt'  # Elindeki 10000x1 verinin adı (txt veya csv)

# --- 2. GERÇEK VERİYİ YÜKLE ---
# Eğer dosya txt ise:
try:
    gt_data = np.loadtxt(ground_truth_path)
except:
    print("Veri dosyası okunamadı! Dosya yolunu veya formatı kontrol et.")
    gt_data = np.zeros(1000) # Kod patlamasın diye boş veri

# --- 3. VİDEO BİLGİLERİNİ AL ---
cap = cv.VideoCapture(video_path)
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv.CAP_PROP_FPS)
duration = total_frames / fps

print(f"--- ANALİZ RAPORU ---")
print(f"Video Süresi: {duration:.2f} saniye")
print(f"Video FPS: {fps}")
print(f"Ground Truth Veri Sayısı: {len(gt_data)}")
print(f"Tahmini GT Frekansı: {len(gt_data)/duration:.2f} Hz")

# Eğer frekanslar uyuşmuyorsa uyarı ver
if abs(len(gt_data) - total_frames) > 100:
    print("UYARI: Video karesi sayısı ile veri sayısı uyuşmuyor!")
    print("Grafik çizerken interpolasyon (eşitleme) yapacağız.")

# --- 4. OPTICAL FLOW DÖNGÜSÜ ---
# (Senin çalışan kodun, sadece çizim yerine veri toplama modunda)
feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

estimated_speeds = [] # Bizim sonuçları burada biriktireceğiz

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Nokta azaldıysa yenile
    if p0 is None or len(p0) < 50:
        p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        if p0 is None: # Hala bulamazsa 0 bas
            estimated_speeds.append(0)
            continue
        old_gray = frame_gray.copy()
        continue

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        displacement = good_new - good_old
        dy_list = displacement[:, 1] # Sadece ileri-geri hareket (Y ekseni)
        
        # Medyan al (Gürültü temizliği)
        if len(dy_list) > 0:
            median_dy = np.median(dy_list)
            # Araba ileri gidiyorsa piksel aşağı akar (pozitif), hızı pozitif yapalım
            # Eğer değerler ters çıkarsa buraya eksi koy: -median_dy
            estimated_speeds.append(-median_dy) 
        else:
            estimated_speeds.append(0)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        estimated_speeds.append(0)

cap.release()

# --- 5. SONUÇLARI KARŞILAŞTIR (PLOT) ---
# Veri boyutlarını eşitle (Interpolasyon)
# Videodaki frame sayısı ile elindeki veri sayısı tutmayabilir.
# Hepsini 0 ile 1 arasında normalize edip "Trend" (Eğilim) kıyaslaması yapacağız.

x_gt = np.linspace(0, duration, len(gt_data))
x_est = np.linspace(0, duration, len(estimated_speeds))

# Normalize et (Çünkü birimler farklı: Biri m/s, biri pixel/frame)
# Amaç: ŞEKİL aynı mı onu görmek.
norm_gt = (gt_data - np.min(gt_data)) / (np.max(gt_data) - np.min(gt_data))
norm_est = (estimated_speeds - np.min(estimated_speeds)) / (np.max(estimated_speeds) - np.min(estimated_speeds))

plt.figure(figsize=(12, 6))
plt.plot(x_gt, norm_gt, 'r-', alpha=0.7, label='Ground Truth (Gerçek Veri)')
plt.plot(x_est, norm_est, 'b-', alpha=0.7, label='Optical Flow (Senin Kodun)')
plt.title("Doğrulama Testi: Gerçek vs Tahmin (Normalize Edilmiş)")
plt.xlabel("Zaman (saniye)")
plt.ylabel("Normalize Hız (0-1 Arası)")
plt.legend()
plt.grid(True)
plt.show()
