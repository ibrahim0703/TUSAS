class StereoRectifier:
    def __init__(self):
        self.image_size = (512, 512)
        
        # Kamera Matrisleri ve Balıkgözü (Equidistant) Katsayıları
        self.K_left = np.array([[190.978, 0.0, 254.931], [0.0, 190.973, 256.897], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_left = np.array([0.003482, 0.000715, -0.002053, 0.0002029], dtype=np.float64)
        
        self.K_right = np.array([[190.442, 0.0, 252.597], [0.0, 190.434, 254.917], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_right = np.array([0.003400, 0.001766, -0.002663, 0.0003299], dtype=np.float64)

        self.R = np.array([
            [ 0.99999719,  0.00160241,  0.00174676],
            [-0.00160269,  0.9999987 ,  0.00016067],
            [-0.0017465 , -0.00016347,  0.99999846]
        ], dtype=np.float64)
        self.T = np.array([-0.10093155, -0.00017163, -0.00067332], dtype=np.float64)

        # İŞTE MÜHENDİSLİK BURADA: balance=0.0 ile OpenCV'nin en güvenli odak uzaklığını KENDİSİNİN bulmasını sağlıyoruz.
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            self.K_left, self.D_left, self.K_right, self.D_right,
            self.image_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            balance=0.0 # Siyah ve yırtık pikselleri acımasızca kes!
        )

        # Yeni hesaplanan güvenli Odak Uzaklığını (f) Z hesabında kullanmak üzere kaydediyoruz.
        self.f_new = P1[0, 0] 

        # Haritaları oluştur
        self.map1_l, self.map2_l = cv2.fisheye.initUndistortRectifyMap(self.K_left, self.D_left, R1, P1, self.image_size, cv2.CV_16SC2)
        self.map1_r, self.map2_r = cv2.fisheye.initUndistortRectifyMap(self.K_right, self.D_right, R2, P2, self.image_size, cv2.CV_16SC2)

    def process(self, img_left, img_right):
        clean_left = cv2.remap(img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        clean_right = cv2.remap(img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return clean_left, clean_right
    # MATEMATİKSEL KANIT (DERİNLİK - Z)
    d = good_left[0][0] - good_right[0][0] 
    if d > 0:
        # rectifier.f_new OpenCV'nin bulduğu KUSURSUZ odak uzaklığıdır.
        z = (rectifier.f_new * 0.1009) / d
        print(f"[MATEMATİK] Yeni güvenli odak uzaklığı (f): {rectifier.f_new:.2f}")
        print(f"[MATEMATİK] İlk noktanın piksel kayması (Disparity): {d:.2f} piksel")
        print(f"[MATEMATİK] İlk noktanın kameraya uzaklığı (Derinlik): {z:.2f} metre")
