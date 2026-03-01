import cv2
import numpy as np

class StereoRectifier:
    def __init__(self):
        # 1. İç Parametreler (Camera1 ve Camera2'den)
        self.K_left = np.array([[190.978, 0.0, 254.931], [0.0, 190.973, 256.897], [0.0, 0.0, 1.0]])
        self.D_left = np.array([0.003482, 0.000715, -0.002053, 0.0002029])
        
        self.K_right = np.array([[190.442, 0.0, 252.597], [0.0, 190.434, 254.917], [0.0, 0.0, 1.0]])
        self.D_right = np.array([0.003400, 0.001766, -0.002663, 0.0003299])

        # 2. Dış Parametreler (camchain.yaml'dan T_cn_cnm1)
        # Sağ kameranın Sol kameraya göre konumu (R ve T)
        self.R = np.array([
            [ 0.99999719,  0.00160241,  0.00174676],
            [-0.00160269,  0.9999987 ,  0.00016067],
            [-0.0017465 , -0.00016347,  0.99999846]
        ])
        self.T = np.array([-0.10093155, -0.00017163, -0.00067332]) # Baseline burada gizli (~10cm)

        self.image_size = (512, 512)
        
        # 3. Rectification Matrislerini Sadece 1 Kere Hesapla (CPU tasarrufu)
        self._init_rectification_maps()

    def _init_rectification_maps(self):
        # Fisheye Stereo Rectify (Sanal paralel düzlemleri oluşturur)
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            self.K_left, self.D_left, self.K_right, self.D_right, 
            self.image_size, self.R, self.T, 
            cv2.CALIB_ZERO_DISPARITY, (0, 0)
        )
        
        # Piksellerin yeni yerlerinin haritasını çıkar
        self.map1_l, self.map2_l = cv2.fisheye.initUndistortRectifyMap(self.K_left, self.D_left, R1, P1, self.image_size, cv2.CV_16SC2)
        self.map1_r, self.map2_r = cv2.fisheye.initUndistortRectifyMap(self.K_right, self.D_right, R2, P2, self.image_size, cv2.CV_16SC2)

    def process(self, img_left, img_right):
        """
        Girdi: Ham sol ve sağ resimler
        Çıktı: Bükülmeleri giderilmiş ve Y-ekseninde kusursuz hizalanmış resimler
        """
        rect_left = cv2.remap(img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return rect_left, rect_right
