import cv2
import numpy as np
from config import k_left, d_left, k_right, d_right, r_matrix, t_vector, baseline

class IMUPreintegrator:
    """
    Basit IMU pre-integrasyon sınıfı.
    İki kare arasındaki ivme + açısal hız verilerini entegre eder.
    Sonuç: delta_R (rotasyon tahmini), delta_v (hız değişimi), delta_p (konum değişimi)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.delta_R = np.eye(3)       # Rotasyon tahmini (3x3)
        self.delta_v = np.zeros(3)     # Hız değişimi (m/s)
        self.delta_p = np.zeros(3)     # Konum değişimi (m)
        self.dt_total = 0.0

    def integrate(self, imu_data):
        """
        imu_data: list of dict -> her biri:
          {'dt': float, 'gyro': [wx,wy,wz] rad/s, 'accel': [ax,ay,az] m/s^2}
        Yerçekimi vektörü kamera çerçevesinde (yaklaşık) aşağıya doğru tanımlıdır.
        TUM-VI için IMU gravity = ~9.81 m/s^2 Z ekseni
        """
        gravity = np.array([0.0, 0.0, -9.81])  # IMU çerçevesinde yerçekimi

        for d in imu_data:
            dt = d['dt']
            gyro = np.array(d['gyro'])
            accel = np.array(d['accel'])

            # Rodrigues formülü ile küçük rotasyon
            angle = np.linalg.norm(gyro) * dt
            if angle > 1e-8:
                axis = gyro / np.linalg.norm(gyro)
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            else:
                dR = np.eye(3)

            # Konum ve hız güncelleme (mid-point entegrasyon)
            accel_world = self.delta_R @ accel - gravity
            self.delta_p += self.delta_v * dt + 0.5 * accel_world * dt ** 2
            self.delta_v += accel_world * dt
            self.delta_R = self.delta_R @ dR
            self.dt_total += dt

    def get_pose_prior(self):
        """
        IMU'dan gelen rotasyon tahminini rvec olarak döndür.
        PnP için initial guess olarak kullanılır.
        """
        rvec, _ = cv2.Rodrigues(self.delta_R)
        tvec_prior = self.delta_p.reshape(3, 1)
        return rvec, tvec_prior


class StereoOdometryTracker:
    def __init__(self):
        # Görüntü boyutu — TUM-VI: 512x512
        self.img_size = (512, 512)

        # ORB: daha fazla nokta başlat, filtreleme sonrası yeterli kalsın
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Stereo rectification (fisheye)
        # DİKKAT: cv2.fisheye.stereoRectify 7 değer döndürür
        (self.R1, self.R2,
         self.P1, self.P2,
         self.Q, _, _) = cv2.fisheye.stereoRectify(
            k_left, d_left,
            k_right, d_right,
            self.img_size,
            r_matrix, t_vector,
            flags=cv2.CALIB_ZERO_DISPARITY,
            balance=0.0,          # tüm piksel geçerli alanda
            fov_scale=1.0
        )

        # Rectify edilmiş ideal focal length
        self.f_ideal = self.P1[0, 0]
        self.cx = self.P1[0, 2]
        self.cy = self.P1[1, 2]

        # Rectify map'leri (remap için)
        self.map1_left, self.map2_left = cv2.fisheye.initUndistortRectifyMap(
            k_left, d_left, self.R1, self.P1, self.img_size, cv2.CV_16SC2
        )
        self.map1_right, self.map2_right = cv2.fisheye.initUndistortRectifyMap(
            k_right, d_right, self.R2, self.P2, self.img_size, cv2.CV_16SC2
        )

        # SGBM parametreleri (dense disparity)
        win_size = 11
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,       # 64 piksel max disparity
            blockSize=win_size,
            P1=8 * 1 * win_size ** 2,
            P2=32 * 1 * win_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # IMU preintegrator
        self.imu = IMUPreintegrator()

        # İdeal pinhole K (rectify sonrası, distortion yok)
        self.K_rect = self.P1[:3, :3]

    def rectify_images(self, raw_left, raw_right):
        """Ham fisheye görüntüleri rectify et."""
        rect_left = cv2.remap(raw_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(raw_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        return rect_left, rect_right

    def process_space_get_depth(self, raw_left, raw_right):
        """
        Stereo çiftten 3B nokta bulutu çıkar.
        Döner: (raw_left üzerindeki 2D noktalar, 3D noktalar)
        """
        rect_left, rect_right = self.rectify_images(raw_left, raw_right)

        # --- YÖNTEM 1: SGBM dense disparity (daha güvenilir) ---
        disp_map = self.sgbm.compute(rect_left, rect_right).astype(np.float32) / 16.0

        # Rectify görüntü üzerinde feature noktası bul
        kp, des = self.orb.detectAndCompute(rect_left, None)
        if des is None or len(kp) == 0:
            return [], [], rect_left

        valid_2d_rect = []   # rectify uzayındaki 2D koordinatlar
        valid_2d_raw = []    # ham görüntüdeki 2D koordinatlar (LK flow için)
        valid_3d = []        # 3D noktalar

        for k in kp:
            x, y = k.pt
            xi, yi = int(x), int(y)

            # Görüntü sınırı kontrolü
            if xi < 1 or xi >= self.img_size[0] - 1 or yi < 1 or yi >= self.img_size[1] - 1:
                continue

            # 3x3 komşulukta medyan disparity (gürültüye karşı)
            patch = disp_map[max(0,yi-1):yi+2, max(0,xi-1):xi+2]
            disp = float(np.median(patch))

            # Disparity geçerlilik: minimum 1.0 piksel
            if disp < 1.0:
                continue

            z = (self.f_ideal * baseline) / disp

            # TUM-VI room: 0.3m - 15m arası geçerli
            if not (0.3 < z < 15.0):
                continue

            X = (x - self.cx) * z / self.f_ideal
            Y = (y - self.cy) * z / self.f_ideal

            valid_2d_rect.append([x, y])
            valid_3d.append([X, Y, z])

            # Ham görüntüdeki karşılığı için ters undistort yap
            # (LK flow ham görüntü üzerinde çalışacak)
            raw_pt = cv2.fisheye.distortPoints(
                np.array([[[x, y]]], dtype=np.float32),
                k_left, d_left
            )
            valid_2d_raw.append(raw_pt[0][0])

        if len(valid_3d) == 0:
            return [], [], rect_left

        pts_raw = np.array(valid_2d_raw, dtype=np.float32).reshape(-1, 1, 2)
        pts_3d = np.array(valid_3d, dtype=np.float32)
        return pts_raw, pts_3d, rect_left

    def track_time_get_flow(self, img_t0, img_t1, pts_t0_raw):
        """
        LK Optical Flow ile noktaları t0'dan t1'e takip et.
        Ham (distorted) görüntüler üzerinde çalışır.
        """
        if len(pts_t0_raw) == 0:
            return [], [], []

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        pts_input = np.array(pts_t0_raw, dtype=np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(img_t0, img_t1, pts_input, None, **lk_params)

        # Geri doğrulama (backward check) — sahte eşleşmeleri eler
        p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(img_t1, img_t0, p1, None, **lk_params)
        back_error = np.abs(pts_input - p0_back).reshape(-1, 2).max(axis=1)

        valid_p0_raw = []
        valid_p1_raw = []
        movements = []

        for i in range(len(pts_input)):
            if st[i] == 1 and st_back[i] == 1 and back_error[i] < 1.0:
                x0, y0 = pts_input[i].ravel()
                x1, y1 = p1[i].ravel()
                valid_p0_raw.append((x0, y0))
                valid_p1_raw.append((x1, y1))
                movements.append(np.hypot(x1 - x0, y1 - y0))

        return valid_p0_raw, valid_p1_raw, movements

    def undistort_points_for_pnp(self, pts_raw):
        """
        Ham (distorted) 2D noktaları fisheye undistort ile
        rectify edilmiş ideal pinhole uzayına çevir.
        PnP'ye bu noktalar ve K_rect verilmeli.
        """
        pts = np.array(pts_raw, dtype=np.float32).reshape(-1, 1, 2)
        # R=R1: rektifikasyon rotasyonu da uygula (3D noktalarla aynı uzay)
        undist = cv2.fisheye.undistortPoints(pts, k_left, d_left, R=self.R1, P=self.P1)
        return undist.reshape(-1, 1, 2)

    def calculate_odometry(self, pts_3d_t0, pts_2d_t1_raw, imu_rvec=None, imu_tvec=None):
        """
        PnP + RANSAC ile kamera hareketi hesapla.
        pts_3d_t0  : rectify uzayındaki 3D noktalar
        pts_2d_t1_raw : t1 karesindeki ham (distorted) 2D noktalar
        imu_rvec/tvec : IMU'dan gelen initial guess (varsa)
        """
        if len(pts_3d_t0) < 6:
            return None, None, None

        # KRİTİK: Ham noktaları rectify uzayına çevir
        pts_2d_undist = self.undistort_points_for_pnp(pts_2d_t1_raw)

        # IMU initial guess kullan (varsa yakınsama hızlanır)
        use_extrinsic = imu_rvec is not None and imu_tvec is not None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_t0.astype(np.float64),
            pts_2d_undist.astype(np.float64),
            self.K_rect,          # rectify edilmiş ideal K
            None,                 # distortion YOK — nokta zaten undistort edildi
            rvec=imu_rvec.astype(np.float64) if use_extrinsic else None,
            tvec=imu_tvec.astype(np.float64) if use_extrinsic else None,
            useExtrinsicGuess=use_extrinsic,
            reprojectionError=2.5,
            confidence=0.999,
            iterationsCount=200,
            flags=cv2.SOLVEPNP_EPNP
        )

        if success and inliers is not None and len(inliers) >= 6:
            # Refine with LM (daha hassas sonuç)
            rvec, tvec = cv2.solvePnPRefineLM(
                pts_3d_t0[inliers.ravel()].astype(np.float64),
                pts_2d_undist[inliers.ravel()].astype(np.float64),
                self.K_rect,
                None,
                rvec, tvec
            )
            return rvec, tvec, inliers

        return None, None, None