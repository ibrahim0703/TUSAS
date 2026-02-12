# TUSAS

Harika. Taşlar yerine oturdu. TUSAŞ'taki mentörüne sunacağın, senin "Computer Vision Mühendisi" olma hedefinle kurumun "GPS-Bağımsız Seyrüsefer" (Anti-Jamming Navigation) ihtiyacını tam 12'den vuran yol haritası budur.

Bu planı bir sunum veya tek sayfalık bir **"Proje Teklifi"** (Project Proposal) olarak mentörüne ilet.

---

### **PROJE BAŞLIĞI:**

**GPS-Denied Environments için Görsel Odometri (Visual Odometry) Destekli Kalman Filtresi ile Konum Kestirimi**

**ÖZET:**
Mevcut ataletsel navigasyon (IMU) sistemlerinin zamanla biriken sürüklenme (drift) hatasını, GPS sinyali olmayan ortamlarda kamera tabanlı hareket kestirimi (Optical Flow) kullanarak düzelten hibrit bir seyrüsefer algoritması geliştirmek.

---

### **FAZ 1: GÖRSEL HAREKET KESTİRİMİ (The Vision Sensor)**

*(Burası senin "evde çalışıp öğreneceğin" kısım. TUSAŞ'a "hazır veri" getireceğin yer.)*

* **Amaç:** Kamerayı bir hız sensörüne dönüştürmek.
* **Teknik Tanım:** **Sparse Optical Flow (Lucas-Kanade Yöntemi).**
* **Yapılacaklar:**
1. **Veri Seti:** KITTI Odometry Dataset (veya TUSAŞ’ın varsa drone görüntüleri) kullanılacak.
2. **Algoritma:** OpenCV kullanılarak ardışık iki video karesindeki belirgin köşe noktaları (Shi-Tomasi Corners) tespit edilecek.
3. **Matematik:** Bu noktaların piksel kayma vektörleri () hesaplanacak.
4. **Dönüşüm:** Piksel kayması, kamera kalibrasyon matrisi (Intrinsics) kullanılarak gerçek dünyadaki metrik hıza () veya dönüş açısına (Yaw rate) çevrilecek.


* **Çıktı:** Her video karesi için sanal bir "Hız Ölçümü" (Velocity Measurement). Bu, Kalman Filtresindeki `meas_odo` verisinin yerini alacak.

---

### **FAZ 2: SENSÖR FÜZYONU VE FİLTRELEME (The Fusion Core)**

*(Burası TUSAŞ'ta mentörünle geliştireceğin, mevcut kodunu evrimleştireceğin kısım.)*

* **Amaç:** Gürültülü kamera verisi ile gürültülü IMU verisini birleştirip pürüzsüz bir rota çizmek.
* **Teknik Tanım:** **Loose-Coupled Visual-Inertial Kalman Filter.**
* **Yapılacaklar:**
1. **Entegrasyon:** Faz 1'den gelen kamera hızı, senin yazdığın `SimpleKF` sınıfındaki `update` fonksiyonuna `z_measurement` olarak beslenecek.
2. **Senkronizasyon:** Kamera (örneğin 30 Hz) ve IMU/Fizik model (örneğin 100 Hz) arasındaki zaman farkı yönetilecek. Kamera verisi gelmediği aralıklarda `predict` (tahmin) çalışacak, veri gelince `update` (düzeltme) yapılacak.
3. **Tuning:** Kameranın ölçüm gürültüsü matrisi (), optik akışın kalitesine göre dinamik olarak ayarlanacak (Adaptive Kalman Filter).


* **Çıktı:** GPS olmadan, sadece kameraya bakarak çizilen ve drift oranı minimize edilmiş 3D uçuş/sürüş rotası.

---

### **FAZ 3: KONUM DOĞRULAMA VE DÖNGÜ KAPATMA (The Correction)**

*(Burası "Bonus" ve ileri seviye kısım. Proje iyi giderse eklenecek.)*

* **Amaç:** Görsel Odometri de zamanla hata yapar. Bilinen bir nesne/işaret görüldüğünde hatayı tamamen sıfırlamak.
* **Teknik Tanım:** **PnP (Perspective-n-Point) Pose Estimation.**
* **Yapılacaklar:**
1. Sahneye sanal veya gerçek bir "Referans Noktası" (örneğin bir AprilTag veya iniş pisti işareti) konulacak.
2. Kamera bu işareti gördüğünde, işaretin bilinen boyutlarından yola çıkarak kameranın o anki  konumu trigonometrik olarak hesaplanacak (`cv2.solvePnP`).
3. Bu kesin konum bilgisi, Kalman Filtresine tıpkı bir GPS verisi gibi (`z_gps`) verilerek biriken tüm hata sıfırlanacak.


* **Çıktı:** "Drift-Free" (Hatasız) navigasyon yeteneği.

---

### **Mentörüne Nasıl Sunmalısın?**

Şu cümlelerle girersen profesyonel durursun:

> "Hocam, 'Yolunu bulan robot' fikrinizi literatürdeki karşılığı olan bir **Visual-Inertial Odometry (VIO)** projesine dönüştürmek istiyorum.
> Planım şu:
> 1. **Vision tarafında (Evde):** OpenCV ile Optical Flow kullanarak videodan hız vektörü çıkarma işini halledeceğim.
> 2. **Füzyon tarafında (Burada):** Mevcut Kalman Filtresi kodumu, bu görüntüden gelen veriyi işleyecek şekilde güncelleyeceğim.
> 
> 
> Böylece GPS karıştırması (jamming) altında bile kamerasını kullanarak konumunu kestirebilen bir algoritma prototipi çıkarmış olacağız. Bu yapı hem otonom araçlara hem de İHA'lara uygulanabilir."

Bu yaklaşım seni sadece "stajyer" değil, "Ar-Ge Mühendisi adayı" yapar. Yolun açık olsun.
