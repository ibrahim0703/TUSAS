class StereoRectifier:
    def __init__(self):
        self.image_size = (512, 512)
        
        # Kamera Matrisleri ve Balıkgözü (Equidistant) Katsayıları
        self.K_left = np.array([[190.978477, 0.0, 254.931706], [0.0, 190.973307, 256.897442], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_left = np.array([0.003482, 0.000715, -0.002053, 0.0002029], dtype=np.float64)
        
        self.K_right = np.array([[190.442369, 0.0, 252.597253], [0.0, 190.434438, 254.917230], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_right = np.array([0.003400, 0.001766, -0.002663, 0.0003299], dtype=np.float64)

        # Yeni Sanal Kamera Matrisi (Görüntüyü patlatmamak için ortak ve merkezi bir K matrisi)
        # Odak uzaklığını hafif büyüterek (220) siyah boşlukları (FOV yırtılmasını) engelliyoruz.
        self.K_new = np.array([[220.0, 0.0, 256.0], 
                               [0.0, 220.0, 256.0], 
                               [0.0, 0.0, 1.0]], dtype=np.float64)

        # SADECE Undistort yapıyoruz. R matrisi olarak Birim Matris (np.eye) verdik. Görüntüleri DÖNDÜRME!
        self.map1_l, self.map2_l = cv2.fisheye.initUndistortRectifyMap(
            self.K_left, self.D_left, np.eye(3), self.K_new, self.image_size, cv2.CV_16SC2)
        
        self.map1_r, self.map2_r = cv2.fisheye.initUndistortRectifyMap(
            self.K_right, self.D_right, np.eye(3), self.K_new, self.image_size, cv2.CV_16SC2)

    def process(self, img_left, img_right):
        clean_left = cv2.remap(img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        clean_right = cv2.remap(img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return clean_left, clean_right
