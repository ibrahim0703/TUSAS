"""
c_tracker.py  —  Stereo Visual-Inertial Odometry
=================================================
Drone anlık hız tahmini için optimize edilmiş.

State vektörü (9D Kalman):
    x = [px, py, pz,   vx, vy, vz,   bg_x, bg_y, bg_z]
         pozisyon(m)   hız(m/s)       gyro bias(rad/s)

Predict → IMU preintegration (bias-aware, Rodrigues)
Update  → PnP tvec ölçümü (SQPNP)
"""

import cv2
import numpy as np
from config import k_left, d_left, k_right, d_right, r_matrix, t_vector, baseline


# ─────────────────────────────────────────────────────────────────────────────
# 9D KALMAN FİLTRESİ
# ─────────────────────────────────────────────────────────────────────────────
class KalmanVIO:
    """
    State: [px, py, pz, vx, vy, vz, bgx, bgy, bgz]  (9x1)

    Parametreler (sensör kalitesine göre ayarla):
        sigma_accel  : ivmeölçer gürültüsü (m/s²)
        sigma_gyro   : jiroskop gürültüsü (rad/s)
        sigma_bias   : bias random-walk (rad/s / sqrt(s))
        sigma_vision : PnP tvec ölçüm gürültüsü (m)
    """

    def __init__(self,
                 sigma_accel:  float = 0.0396,     # accel_noise_density(0.0028) * sqrt(200 Hz)
                 sigma_gyro:   float = 0.00226,    # gyro_noise_density(0.00016) * sqrt(200 Hz)
                 sigma_bias:   float = 0.00000156, # gyro_random_walk(0.000022)  / sqrt(200 Hz)
                 sigma_vision: float = 0.02):

        self.n = 9

        # ── Başlangıç state & kovaryans ──────────────────────────────────────
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 0.1
        self.P[6:9, 6:9] = np.eye(3) * 1e-4   # bias başlangıçta küçük belirsizlik

        # ── Gürültü sabitleri ─────────────────────────────────────────────────
        self.sa = sigma_accel
        self.sg = sigma_gyro
        self.sb = sigma_bias
        self.sv = sigma_vision

        # ── Ölçüm matrisi H — sadece hız gözlemlenir (indeks 3,4,5) ─────────
        # PnP bize tvec = kamera yer değiştirmesi → dt'ye bölerek hız elde edilir
        self.H = np.zeros((3, self.n))
        self.H[0, 3] = 1.0   # vx
        self.H[1, 4] = 1.0   # vy
        self.H[2, 5] = 1.0   # vz

        self.R_meas = np.eye(3) * self.sv ** 2

        self._initialized = False

    # ── İlk ölçümle başlat ───────────────────────────────────────────────────
    def initialize(self, v0: np.ndarray):
        self.x[3:6] = v0.ravel()
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ── PREDICT: IMU ile durum öngörüsü ──────────────────────────────────────
    def predict(self, accel_world: np.ndarray, dt: float):
        """
        accel_world : dünya çerçevesinde lineer ivme, yerçekimi çıkarılmış (m/s²)
        dt          : zaman adımı (s)
        """
        if not self._initialized:
            return

        a = accel_world.ravel()
        dt2 = dt * dt
        dt3 = dt2 * dt

        # State geçiş matrisi F (Sabit hız + ivme modeli)
        F = np.eye(self.n)
        F[0, 3] = dt    # px += vx * dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Proses gürültüsü Q
        Q = np.zeros((self.n, self.n))
        # Pozisyon: ivme gürültüsünden
        Q[0, 0] = 0.25 * dt3 * self.sa ** 2
        Q[1, 1] = Q[0, 0]
        Q[2, 2] = Q[0, 0]
        # Hız: ivme gürültüsünden
        Q[3, 3] = dt * self.sa ** 2
        Q[4, 4] = Q[3, 3]
        Q[5, 5] = Q[3, 3]
        # Bias: random-walk
        Q[6, 6] = dt * self.sb ** 2
        Q[7, 7] = Q[6, 6]
        Q[8, 8] = Q[6, 6]

        # State güncelle
        new_x = F @ self.x
        new_x[0] += 0.5 * a[0] * dt2
        new_x[1] += 0.5 * a[1] * dt2
        new_x[2] += 0.5 * a[2] * dt2
        new_x[3] += a[0] * dt
        new_x[4] += a[1] * dt
        new_x[5] += a[2] * dt

        self.x = new_x
        self.P = F @ self.P @ F.T + Q

    # ── UPDATE: PnP tvec ile ölçüm güncellemesi ───────────────────────────────
    def update(self, tvec: np.ndarray, dt: float, n_inliers: int):
        """
        tvec      : PnP kamera yer değiştirmesi (3,) metre
        dt        : kare süresi (s)
        n_inliers : inlier sayısı → ölçüm güvenilirliği ağırlığı
        """
        if not self._initialized:
            self.initialize(tvec.ravel() / max(dt, 1e-9))
            return

        # tvec → anlık hız tahmini
        v_meas = tvec.ravel() / max(dt, 1e-9)

        # İnlier sayısı arttıkça ölçüme daha çok güven
        inlier_factor = max(1.0, 50.0 / max(n_inliers, 1))
        R_dyn = self.R_meas * inlier_factor

        # Kalman kazancı
        S = self.H @ self.P @ self.H.T + R_dyn
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # İnovasyon
        y = v_meas - self.H @ self.x

        # State & kovaryans güncelle (Joseph form — sayısal kararlılık)
        self.x = self.x + K @ y
        I_KH   = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_dyn @ K.T

    # ── Özellikler ────────────────────────────────────────────────────────────
    @property
    def velocity(self) -> np.ndarray:
        """Anlık hız vektörü (m/s), 3D."""
        return self.x[3:6].copy()

    @property
    def speed(self) -> float:
        """Anlık skaler hız (m/s)."""
        return float(np.linalg.norm(self.x[3:6]))

    @property
    def position(self) -> np.ndarray:
        """Kümülatif pozisyon (m), 3D."""
        return self.x[0:3].copy()

    @property
    def gyro_bias(self) -> np.ndarray:
        """Tahmin edilen gyro bias (rad/s), 3D."""
        return self.x[6:9].copy()


# ─────────────────────────────────────────────────────────────────────────────
# IMU PREINTEGRATİON (bias-aware, Rodrigues)
# ─────────────────────────────────────────────────────────────────────────────
class IMUPreintegrator:
    """
    Bias düzeltmeli IMU preintegration.
    Kalman'dan gelen gyro bias her frame başında set_bias() ile enjekte edilir.
    """

    def __init__(self):
        self.gravity    = np.array([0.0, 0.0, 9.81])
        self.bias_gyro  = np.zeros(3)
        self.reset()

    def set_bias(self, bg: np.ndarray):
        """Kalman state'inden gelen bias güncellemesi."""
        self.bias_gyro = bg.ravel().copy()

    def reset(self):
        self.delta_R  = np.eye(3)
        self.delta_v  = np.zeros(3)
        self.delta_p  = np.zeros(3)
        self.dt_total = 0.0

    def integrate(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        # Bias çıkar
        gyro_c = gyro - self.bias_gyro

        # Rodrigues rotasyon formülü
        angle = np.linalg.norm(gyro_c) * dt
        if angle > 1e-10:
            axis = gyro_c / np.linalg.norm(gyro_c)
            K = np.array([
                [ 0,        -axis[2],  axis[1]],
                [ axis[2],   0,       -axis[0]],
                [-axis[1],   axis[0],  0      ]
            ])
            dR = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        else:
            dR = np.eye(3)

        accel_world   = self.delta_R @ accel - self.gravity
        self.delta_p += self.delta_v * dt + 0.5 * accel_world * dt ** 2
        self.delta_v += accel_world * dt
        self.delta_R   = self.delta_R @ dR
        self.dt_total += dt

    def get_prediction(self):
        """(ΔR, Δv, Δp) döndür."""
        return self.delta_R.copy(), self.delta_v.copy(), self.delta_p.copy()

    def get_accel_world_mean(self) -> np.ndarray:
        """
        Frame aralığındaki ortalama dünya-çerçevesi ivmesi.
        Kalman predict girdisi olarak kullanılır.
        """
        if self.dt_total < 1e-9:
            return np.zeros(3)
        return self.delta_v / self.dt_total


# ─────────────────────────────────────────────────────────────────────────────
# STEREO ODOMETRY TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class StereoOdometryTracker:

    def __init__(self):
        # ── Özellik dedektörü ─────────────────────────────────────────────────
        self.orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=10
        )

        # ── IMU & Kalman ──────────────────────────────────────────────────────
        self.imu    = IMUPreintegrator()

        # TUM-VI imu_config.yaml degerlerinden hesaplandi (200 Hz):
        #   sigma_accel  = accel_noise_density(0.0028)  * sqrt(200) = 0.0396  m/s2
        #   sigma_gyro   = gyro_noise_density(0.00016)  * sqrt(200) = 0.00226 rad/s
        #   sigma_bias   = gyro_random_walk(0.000022)   / sqrt(200) = 1.56e-6 rad/s
        #   sigma_vision = PnP reprojectionError(2px) kaynakli, elle ayarli
        self.kalman = KalmanVIO(
            sigma_accel=0.0396,
            sigma_gyro=0.00226,
            sigma_bias=0.00000156,
            sigma_vision=0.02
        )

        # ── Stereo rektifikasyon matrisleri ───────────────────────────────────
        result = cv2.fisheye.stereoRectify(
            k_left, d_left, k_right, d_right,
            (512, 512), r_matrix, t_vector,
            flags=cv2.CALIB_ZERO_DISPARITY
        )
        self.R1, self.R2, self.P1, self.P2, self.Q = result[:5]

        self.f_ideal = self.P1[0, 0]
        self.K_rect  = self.P1[:3, :3]

        # ── SGBM stereo matcher ───────────────────────────────────────────────
        # numDisparities=128 (64'ten artırıldı: daha iyi yakın mesafe kapsama)
        # blockSize=9       (11'den küçültüldü: daha hızlı, yeterince stabil)
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=9,
            P1=8  * 3 * 9 ** 2,
            P2=32 * 3 * 9 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self._maps_ready = False

    # ── Rektifikasyon haritaları (lazy init) ──────────────────────────────────
    def _ensure_maps(self, h: int, w: int):
        if self._maps_ready:
            return
        self._ml1, self._ml2 = cv2.fisheye.initUndistortRectifyMap(
            k_left,  d_left,  self.R1, self.P1, (w, h), cv2.CV_32FC1)
        self._mr1, self._mr2 = cv2.fisheye.initUndistortRectifyMap(
            k_right, d_right, self.R2, self.P2, (w, h), cv2.CV_32FC1)
        self._maps_ready = True

    def rectify_image(self, img: np.ndarray, side: str = 'left') -> np.ndarray:
        h, w = img.shape[:2]
        self._ensure_maps(h, w)
        maps = (self._ml1, self._ml2) if side == 'left' else (self._mr1, self._mr2)
        return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)

    # ── Stereo → 3D noktalar (vektörize) ─────────────────────────────────────
    def process_space_get_depth(self, raw_left: np.ndarray, raw_right: np.ndarray):
        """
        Disparity haritasından ORB keypoint'lerin 3D konumlarını hesaplar.

        Dönüş:
            pts_rect : (N,1,2) float32  — rectified sol pikseller
            pts_3d   : (N,3)   float32  — kamera çerçevesinde 3D noktalar (m)
            rect_l   : rectified sol görüntü (optical flow için)
        """
        h, w = raw_left.shape[:2]
        self._ensure_maps(h, w)

        rect_l = cv2.remap(raw_left,  self._ml1, self._ml2, cv2.INTER_LINEAR)
        rect_r = cv2.remap(raw_right, self._mr1, self._mr2, cv2.INTER_LINEAR)

        disp = self.sgbm.compute(rect_l, rect_r).astype(np.float32) / 16.0

        kp_list, _ = self.orb.detectAndCompute(rect_l, None)
        if not kp_list:
            return np.array([]), np.array([]), rect_l

        cx = self.P1[0, 2]
        cy = self.P1[1, 2]
        fx = self.P1[0, 0]
        fy = self.P1[1, 1]

        # Tüm keypoint'leri aynı anda işle — for döngüsü yok
        pts = np.array([kp.pt for kp in kp_list], dtype=np.float32)  # (N,2)
        xi  = np.round(pts[:, 0]).astype(int)
        yi  = np.round(pts[:, 1]).astype(int)

        # 1) Görüntü sınırı maskesi
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        pts, xi, yi = pts[valid], xi[valid], yi[valid]

        # 2) Disparity maskesi
        d = disp[yi, xi]
        valid = d >= 0.5
        pts, d = pts[valid], d[valid]

        # 3) Derinlik hesapla & z aralığı filtresi
        z     = (self.f_ideal * baseline) / d
        valid = (z > 0.3) & (z < 20.0)
        pts, z = pts[valid], z[valid]

        if len(pts) == 0:
            return np.array([]), np.array([]), rect_l

        X = (pts[:, 0] - cx) * z / fx
        Y = (pts[:, 1] - cy) * z / fy

        pts_3d   = np.stack([X, Y, z], axis=1).astype(np.float32)
        pts_rect = pts.reshape(-1, 1, 2).astype(np.float32)
        return pts_rect, pts_3d, rect_l

    # ── Optical flow (Lucas-Kanade) ───────────────────────────────────────────
    def track_time_get_flow(self, rect_t0: np.ndarray,
                             rect_t1: np.ndarray,
                             pts_t0:  np.ndarray):
        """
        t0 → t1 arası nokta takibi.

        Dönüş:
            p0_valid  : (K,2) ndarray — takip edilen t0 noktaları
            p1_valid  : (K,2) ndarray — t1'deki karşılıkları
            movements : (K,)  ndarray — piksel hareketi büyüklükleri
        """
        if len(pts_t0) == 0:
            return np.array([]), np.array([]), np.array([])

        lk_params = dict(
            winSize=(15, 15),       # 21→15: TUM-VI için daha uygun pencere
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        p1, st, _ = cv2.calcOpticalFlowPyrLK(rect_t0, rect_t1, pts_t0, None, **lk_params)

        mask     = st.ravel() == 1
        p0_valid = pts_t0[mask].reshape(-1, 2)
        p1_valid = p1[mask].reshape(-1, 2)
        movements = np.hypot(p1_valid[:, 0] - p0_valid[:, 0],
                             p1_valid[:, 1] - p0_valid[:, 1])

        return p0_valid, p1_valid, movements

    # ── 3D-2D eşleştirme (vektörize, for döngüsü yok) ────────────────────────
    def match_3d_2d(self, pts_t0:      np.ndarray,
                    pts_3d:      np.ndarray,
                    p0_arr:      np.ndarray,
                    p1_arr:      np.ndarray,
                    dist_thresh: float = 1.5):
        """
        Optical flow sonucu gelen p0 noktalarını ORB keypoint'leriyle eşleştirip
        karşılık gelen 3D noktaları döndürür.

        Giriş:
            pts_t0      : (N,1,2) ORB keypoint'leri (rectified)
            pts_3d      : (N,3)   ORB keypoint'lerinin 3D karşılıkları
            p0_arr      : (K,2)   flow girdi noktaları
            p1_arr      : (K,2)   flow çıktı noktaları (t1'deki pikseller)
            dist_thresh : eşleştirme için maksimum piksel mesafesi

        Dönüş:
            matched_3d : (M,3) float32
            matched_2d : (M,2) float32
        """
        if len(p0_arr) == 0 or len(pts_3d) == 0:
            return np.array([]), np.array([])

        pts_arr = pts_t0.reshape(-1, 2)   # (N,2)

        # Broadcasting ile (K,N) mesafe matrisi — O(K*N) ama loop yok
        diff  = p0_arr[:, None, :] - pts_arr[None, :, :]  # (K,N,2)
        dists = np.linalg.norm(diff, axis=2)               # (K,N)

        min_idx  = np.argmin(dists, axis=1)                # (K,)
        min_dist = dists[np.arange(len(p0_arr)), min_idx]  # (K,)

        valid = min_dist < dist_thresh
        if not np.any(valid):
            return np.array([]), np.array([])

        matched_3d = pts_3d[min_idx[valid]].astype(np.float32)
        matched_2d = p1_arr[valid].astype(np.float32)
        return matched_3d, matched_2d

    # ── PnP odometry (SQPNP) ─────────────────────────────────────────────────
    def calculate_odometry(self, pts_3d: np.ndarray, pts_2d: np.ndarray):
        """
        SQPNP kullanılır:
          - EPnP'ye kıyasla planar degeneracy'ye daha az duyarlı
          - Drone uçuş gibi farklı derinlik dağılımlarında daha kararlı

        Dönüş:
            rvec    : (3,1) rotasyon vektörü
            tvec    : (3,1) çeviri vektörü (metre)
            inliers : inlier indeks dizisi
        """
        if len(pts_3d) < 6:
            return None, None, None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.reshape(-1, 1, 3),
            pts_2d.reshape(-1, 1, 2),
            self.K_rect,
            None,                            # distortion yok (rectified uzay)
            reprojectionError=2.0,
            confidence=0.999,
            iterationsCount=300,
            flags=cv2.SOLVEPNP_SQPNP        # EPnP → SQPNP
        )

        if success and inliers is not None and len(inliers) >= 6:
            return rvec, tvec, inliers
        return None, None, None

    # ── Kalman adımı: IMU predict + PnP update ────────────────────────────────
    def kalman_step(self, tvec: np.ndarray, dt: float, n_inliers: int):
        """
        1. Güncel Kalman bias tahminini IMU'ya yaz
        2. IMU ortalama ivmesiyle Kalman'ı öngör (predict)
        3. PnP tvec ile Kalman'ı güncelle (update)
        """
        # Bias geri besleme: Kalman → IMU
        self.imu.set_bias(self.kalman.gyro_bias)

        # Ortalama dünya ivmesi → predict
        accel_world = self.imu.get_accel_world_mean()
        self.kalman.predict(accel_world, dt)

        # PnP tvec → update
        self.kalman.update(tvec.ravel(), dt, n_inliers)

    # ── Hız erişimi ───────────────────────────────────────────────────────────
    @property
    def speed_ms(self) -> float:
        """Kalman filtreli anlık skaler hız (m/s)."""
        return self.kalman.speed

    @property
    def velocity_ms(self) -> np.ndarray:
        """Kalman filtreli anlık hız vektörü (m/s), [vx, vy, vz]."""
        return self.kalman.velocity
