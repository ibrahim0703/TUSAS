"""
tracker.py  —  Stereo Visual-Inertial Odometry (Optimized)
============================================================
Drone anlık hız tahmini için optimize edilmiş.
TUM-VI Room1 dataseti için hızlı dönüşlere dayanıklı tasarım.

State vektörü (9D Kalman):
    x = [px, py, pz,   vx, vy, vz,   bg_x, bg_y, bg_z]
         pozisyon(m)   hız(m/s)       gyro bias(rad/s)

Predict → IMU preintegration (bias-aware, Rodrigues)
Update  → PnP tvec ölçümü (SQPNP) + Innovation Gate + Adaptive R

Optimizasyonlar:
  - GFTT + ORB hibrit feature detection
  - Forward-backward optical flow konsistansi
  - IMU-aided rotation compensation (LK initial guess)
  - Mahalanobis innovation gate
  - Düşük inlier frame rejection
  - Adaptif ölçüm gürültüsü
"""

import cv2
import numpy as np
from config import k_left, d_left, k_right, d_right, r_matrix, t_vector, baseline


# ─────────────────────────────────────────────────────────────────────────────
# 9D KALMAN FİLTRESİ (Optimized)
# ─────────────────────────────────────────────────────────────────────────────
class KalmanVIO:
    """
    State: [px, py, pz, vx, vy, vz, bgx, bgy, bgz]  (9x1)

    Parametreler (TUM-VI Room1 için optimize edilmiş):
        sigma_accel  : ivmeölçer gürültüsü (m/s²)
        sigma_gyro   : jiroskop gürültüsü (rad/s)
        sigma_bias   : bias random-walk (rad/s / sqrt(s))
        sigma_vision : PnP tvec ölçüm gürültüsü (m)
        innov_gate   : Mahalanobis innovation gate eşiği (chi-squared 3 DOF)
    """

    def __init__(self,
                 sigma_accel:  float = 0.0396,
                 sigma_gyro:   float = 0.00226,
                 sigma_bias:   float = 0.00000156,
                 sigma_vision: float = 0.015,       # PnP'ye daha fazla güven → underestimation azalır
                 innov_gate:   float = 16.0):        # chi-squared 3 DOF %99.9

        self.n = 9

        # ── Başlangıç state & kovaryans ──────────────────────────────────────
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 0.1
        self.P[3:6, 3:6] = np.eye(3) * 1.0    # Hız başlangıçta belirsiz
        self.P[6:9, 6:9] = np.eye(3) * 1e-4

        # ── Gürültü sabitleri ─────────────────────────────────────────────────
        self.sa = sigma_accel
        self.sg = sigma_gyro
        self.sb = sigma_bias
        self.sv = sigma_vision

        # ── Innovation gate eşiği ────────────────────────────────────────────
        self.innov_gate = innov_gate

        # ── Ölçüm matrisi H — sadece hız gözlemlenir (indeks 3,4,5) ─────────
        self.H = np.zeros((3, self.n))
        self.H[0, 3] = 1.0
        self.H[1, 4] = 1.0
        self.H[2, 5] = 1.0

        self.R_base = np.eye(3) * self.sv ** 2

        self._initialized = False
        self._last_update_accepted = True
        self._consecutive_rejections = 0      # Ardışık red sayısı
        self._MAX_CONSECUTIVE_REJECT = 3      # Bu kadar red sonrası zorla kabul

    # ── İlk ölçümle başlat ───────────────────────────────────────────────────
    def initialize(self, v0: np.ndarray):
        self.x[3:6] = v0.ravel()
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def last_update_accepted(self) -> bool:
        return self._last_update_accepted

    # ── PREDICT: IMU ile durum öngörüsü ──────────────────────────────────────
    def predict(self, accel_world: np.ndarray, dt: float):
        if not self._initialized:
            return

        a = accel_world.ravel()
        dt2 = dt * dt
        dt3 = dt2 * dt

        F = np.eye(self.n)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        Q = np.zeros((self.n, self.n))
        Q[0, 0] = 0.25 * dt3 * self.sa ** 2
        Q[1, 1] = Q[0, 0]
        Q[2, 2] = Q[0, 0]
        Q[3, 3] = dt * self.sa ** 2
        Q[4, 4] = Q[3, 3]
        Q[5, 5] = Q[3, 3]
        Q[6, 6] = dt * self.sb ** 2
        Q[7, 7] = Q[6, 6]
        Q[8, 8] = Q[6, 6]

        new_x = F @ self.x
        new_x[0] += 0.5 * a[0] * dt2
        new_x[1] += 0.5 * a[1] * dt2
        new_x[2] += 0.5 * a[2] * dt2
        new_x[3] += a[0] * dt
        new_x[4] += a[1] * dt
        new_x[5] += a[2] * dt

        # Velocity damping KALDIRILDI — Kalman Q matrisi yeterli belirsizlik ekliyor
        # Eski 0.995 damping her frame'de %0.5 kayıp → 1s'de %22 hız kaybı → underestimation

        self.x = new_x
        self.P = F @ self.P @ F.T + Q

        # Kovaryans tabanı — P'nin aşırı küçülüp ölçümleri reddetmesini engelle
        P_floor = 1e-4
        diag = np.diag(self.P)
        diag_clamped = np.maximum(diag, P_floor)
        np.fill_diagonal(self.P, diag_clamped)

    # ── UPDATE: PnP tvec ile ölçüm güncellemesi + Innovation Gate ─────────
    def update(self, tvec: np.ndarray, dt: float, n_inliers: int):
        if not self._initialized:
            self.initialize(tvec.ravel() / max(dt, 1e-9))
            return

        # tvec → anlık hız tahmini
        v_meas = tvec.ravel() / max(dt, 1e-9)

        # ── Adaptif ölçüm gürültüsü ─────────────────────────────────────────
        # İnlier az → ölçüme daha az güven (daha agresif scaling)
        inlier_factor = max(1.0, 50.0 / max(n_inliers, 1))

        # dt çok büyükse (frame drop) güvenilirlik düşür
        if dt > 0.08:  # >80ms → muhtemelen frame drop
            inlier_factor *= 2.0

        R_dyn = self.R_base * inlier_factor

        # ── Innovation hesapla ───────────────────────────────────────────────
        y = v_meas - self.H @ self.x

        # ── Adaptif Innovation Gate — frame drop'ta gate genişlet ────────────
        effective_gate = self.innov_gate
        if dt > 0.06:  # >60ms → predict belirsizliği yüksek
            effective_gate *= min(dt / 0.05, 3.0)  # max 3x genişleme

        # ── Innovation Gate (Mahalanobis mesafesi) ───────────────────────────
        S = self.H @ self.P @ self.H.T + R_dyn
        S_inv = np.linalg.inv(S)
        mahal_sq = float(y.T @ S_inv @ y)

        if mahal_sq > effective_gate:
            self._consecutive_rejections += 1
            # Kovaryansı şişir — ne kadar çok ardışık red, o kadar agresif
            inflate = 0.1 * self._consecutive_rejections
            self.P[3:6, 3:6] += np.eye(3) * inflate

            # Çok fazla ardışık red → zorla kabul et (filtre kilitlenmesini önle)
            if self._consecutive_rejections < self._MAX_CONSECUTIVE_REJECT:
                self._last_update_accepted = False
                return
            # else: devam et, aşağıda update yapılacak

        self._last_update_accepted = True
        self._consecutive_rejections = 0

        # ── Kalman kazancı ───────────────────────────────────────────────────
        K = self.P @ self.H.T @ S_inv

        # ── State & kovaryans güncelle (Joseph form) ─────────────────────────
        self.x = self.x + K @ y
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_dyn @ K.T

    # ── Predict-only step (ölçüm olmadan) ────────────────────────────────────
    def predict_only(self, dt: float):
        """PnP yok → sabit hız modeli, hafif sönümleme."""
        if not self._initialized:
            return
        self.predict(np.zeros(3), dt)
        # Sadece update yok iken hafif damping (50Hz'de 1s'de %90 koruma)
        DECAY = 0.998
        self.x[3:6] *= DECAY

    # ── Özellikler ────────────────────────────────────────────────────────────
    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[3:6]))

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def gyro_bias(self) -> np.ndarray:
        return self.x[6:9].copy()


# ─────────────────────────────────────────────────────────────────────────────
# IMU PREINTEGRATİON (bias-aware, Rodrigues)
# ─────────────────────────────────────────────────────────────────────────────
class IMUPreintegrator:

    def __init__(self):
        self.gravity    = np.array([0.0, 0.0, 9.81])
        self.bias_gyro  = np.zeros(3)
        self.reset()

    def set_bias(self, bg: np.ndarray):
        self.bias_gyro = bg.ravel().copy()

    def reset(self):
        self.delta_R  = np.eye(3)
        self.delta_v  = np.zeros(3)
        self.delta_p  = np.zeros(3)
        self.dt_total = 0.0

    def integrate(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        gyro_c = gyro - self.bias_gyro

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
        self.delta_R  = self.delta_R @ dR
        self.dt_total += dt

    def get_prediction(self):
        return self.delta_R.copy(), self.delta_v.copy(), self.delta_p.copy()

    def get_accel_world_mean(self) -> np.ndarray:
        if self.dt_total < 1e-9:
            return np.zeros(3)
        return self.delta_v / self.dt_total


# ─────────────────────────────────────────────────────────────────────────────
# STEREO ODOMETRY TRACKER (Optimized)
# ─────────────────────────────────────────────────────────────────────────────
class StereoOdometryTracker:

    # Düşük inlier eşiği — bu değerin altında PnP sonucu kullanılmaz
    MIN_INLIERS_FOR_UPDATE = 20

    def __init__(self):
        # ── ORB dedektör ─────────────────────────────────────────────────────
        self.orb = cv2.ORB_create(
            nfeatures=1500,         # 1000 → 1500: daha fazla feature → daha iyi coverage
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=8         # 10 → 8: daha hassas → daha fazla feature
        )

        # ── GFTT parametreleri (ORB ile birleştirilecek) ─────────────────────
        self.gftt_params = dict(
            maxCorners=800,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )

        # ── IMU & Kalman ──────────────────────────────────────────────────────
        self.imu    = IMUPreintegrator()
        self.kalman = KalmanVIO(
            sigma_accel=0.0396,        # imu_config.yaml: 0.0028 * sqrt(200)
            sigma_gyro=0.00226,        # imu_config.yaml: 0.00016 * sqrt(200)
            sigma_bias=0.00000156,     # imu_config.yaml: 0.000022 / sqrt(200)
            sigma_vision=0.015,        # PnP'ye daha fazla güven
            innov_gate=16.0            # chi-squared 3 DOF %99.9
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

        # ── SGBM stereo matcher (iyileştirilmiş) ─────────────────────────────
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=9,
            P1=8  * 3 * 9 ** 2,
            P2=32 * 3 * 9 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=5,       # 10 → 5: daha fazla valid disparity
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self._maps_ready = False

        # ── Önceki frame hızı (süreklilik kontrolü) ──────────────────────────
        self._prev_speed = 0.0

        # ── EMA çıkış filtresi ────────────────────────────────────────────────
        self._ema_velocity = np.zeros(3)
        self._ema_initialized = False
        self._ema_alpha = 0.6   # Daha hızlı tepki, daha az gecikme

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

    # ── Hibrit Feature Detection: GFTT + ORB ─────────────────────────────────
    def _detect_features_hybrid(self, rect_img: np.ndarray):
        """
        GFTT ve ORB noktalarını birleştir.
        GFTT daha stabil köşeler verir (rotasyona dayanıklı).
        ORB daha fazla ve çeşitli noktalar verir.
        """
        h, w = rect_img.shape[:2]

        # 1) ORB keypoints
        kp_orb, _ = self.orb.detectAndCompute(rect_img, None)
        pts_orb = np.array([kp.pt for kp in kp_orb], dtype=np.float32) if kp_orb else np.empty((0, 2), dtype=np.float32)

        # 2) GFTT keypoints
        gftt = cv2.goodFeaturesToTrack(rect_img, **self.gftt_params)
        pts_gftt = gftt.reshape(-1, 2).astype(np.float32) if gftt is not None else np.empty((0, 2), dtype=np.float32)

        if len(pts_orb) == 0 and len(pts_gftt) == 0:
            return np.empty((0, 2), dtype=np.float32)

        if len(pts_orb) == 0:
            return pts_gftt
        if len(pts_gftt) == 0:
            return pts_orb

        # 3) Birleştir ve tekrarları çıkar (minimum mesafe = 3px)
        all_pts = np.vstack([pts_orb, pts_gftt])
        return self._deduplicate_points(all_pts, min_dist=3.0)

    @staticmethod
    def _deduplicate_points(pts: np.ndarray, min_dist: float = 3.0) -> np.ndarray:
        """Birbirine çok yakın noktaları çıkar (greedy)."""
        if len(pts) <= 1:
            return pts

        # Daha hızlı deduplikasyon: KD-tree benzeri grid yaklaşımı
        keep = np.ones(len(pts), dtype=bool)
        cell_size = min_dist
        cells = {}

        for i in range(len(pts)):
            if not keep[i]:
                continue
            cx, cy = int(pts[i, 0] / cell_size), int(pts[i, 1] / cell_size)
            # Komşu hücrelerde kontrol et
            found_neighbor = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key in cells:
                        for j in cells[key]:
                            dist = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                            if dist < min_dist:
                                keep[i] = False
                                found_neighbor = True
                                break
                    if found_neighbor:
                        break
                if found_neighbor:
                    break

            if keep[i]:
                key = (cx, cy)
                if key not in cells:
                    cells[key] = []
                cells[key].append(i)

        return pts[keep]

    # ── Stereo → 3D noktalar (Hibrit features ile) ───────────────────────────
    def process_space_get_depth(self, raw_left: np.ndarray, raw_right: np.ndarray):
        """
        Hibrit feature detection + disparity → 3D noktalar.

        Dönüş:
            pts_rect : (N,1,2) float32  — rectified sol pikseller
            pts_3d   : (N,3)   float32  — kamera çerçevesinde 3D noktalar (m)
            rect_l   : rectified sol görüntü
        """
        h, w = raw_left.shape[:2]
        self._ensure_maps(h, w)

        rect_l = cv2.remap(raw_left,  self._ml1, self._ml2, cv2.INTER_LINEAR)
        rect_r = cv2.remap(raw_right, self._mr1, self._mr2, cv2.INTER_LINEAR)

        disp = self.sgbm.compute(rect_l, rect_r).astype(np.float32) / 16.0

        # Hibrit feature detection
        pts = self._detect_features_hybrid(rect_l)

        if len(pts) == 0:
            return np.array([]), np.array([]), rect_l

        cx = self.P1[0, 2]
        cy = self.P1[1, 2]
        fx = self.P1[0, 0]
        fy = self.P1[1, 1]

        xi = np.round(pts[:, 0]).astype(int)
        yi = np.round(pts[:, 1]).astype(int)

        # 1) Görüntü sınırı maskesi
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        pts, xi, yi = pts[valid], xi[valid], yi[valid]

        # 2) Disparity maskesi (1.0'a yükseltildi → çok uzak gürültülü noktaları at)
        d = disp[yi, xi]
        valid = d >= 1.0    # 0.5 → 1.0
        pts, d = pts[valid], d[valid]

        # 3) Derinlik hesapla & z aralığı filtresi
        z     = (self.f_ideal * baseline) / d
        valid = (z > 0.3) & (z < 15.0)   # 20m → 15m: uzak noktalar gürültülü
        pts, z = pts[valid], z[valid]

        if len(pts) == 0:
            return np.array([]), np.array([]), rect_l

        X = (pts[:, 0] - cx) * z / fx
        Y = (pts[:, 1] - cy) * z / fy

        pts_3d   = np.stack([X, Y, z], axis=1).astype(np.float32)
        pts_rect = pts.reshape(-1, 1, 2).astype(np.float32)
        return pts_rect, pts_3d, rect_l

    # ── IMU-Aided Rotation Compensation for Optical Flow ──────────────────────
    def _compute_rotation_initial_guess(self, pts_t0: np.ndarray) -> np.ndarray:
        """
        IMU preintegration'dan gelen delta_R kullanarak t0 noktalarının
        t1'deki tahmini konumlarını hesapla.

        Giriş:  pts_t0 (N,1,2) or (N,2)
        Dönüş:  pts_predicted (N,1,2) float32
        """
        delta_R = self.imu.delta_R

        # Eğer rotasyon çok küçükse (identity'ye yakın), compensasyon gerekmez
        angle = np.arccos(np.clip((np.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0))
        if angle < 0.001:  # < 0.057°
            return pts_t0.reshape(-1, 1, 2).astype(np.float32)

        pts = pts_t0.reshape(-1, 2)
        cx = self.P1[0, 2]
        cy = self.P1[1, 2]
        fx = self.P1[0, 0]
        fy = self.P1[1, 1]

        # Piksel → normalize kamera koordinatları
        x_norm = (pts[:, 0] - cx) / fx
        y_norm = (pts[:, 1] - cy) / fy
        ones   = np.ones(len(pts))
        rays   = np.stack([x_norm, y_norm, ones], axis=1)  # (N, 3)

        # Rotasyonu uygula (kamera hareketi → noktaların tersi yönde hareket)
        # delta_R: body frame R, K_rect ile kamera frame'e çevir
        rays_rotated = (delta_R @ rays.T).T  # (N, 3)

        # Normalize kamera → piksel
        z_r = rays_rotated[:, 2]
        z_r = np.where(np.abs(z_r) < 1e-6, 1e-6, z_r)
        x_pred = fx * (rays_rotated[:, 0] / z_r) + cx
        y_pred = fy * (rays_rotated[:, 1] / z_r) + cy

        pts_pred = np.stack([x_pred, y_pred], axis=1).astype(np.float32)
        return pts_pred.reshape(-1, 1, 2)

    # ── Forward-Backward Optical Flow ─────────────────────────────────────────
    def track_time_get_flow(self, rect_t0: np.ndarray,
                             rect_t1: np.ndarray,
                             pts_t0:  np.ndarray):
        """
        Forward-backward konsistans kontrollü optical flow.

        t0 → t1 takip et, sonra t1 → t0 geri takip et.
        İleri-geri mesafesi < 1.0 px olanları kabul et.

        Dönüş:
            p0_valid  : (K,2) ndarray
            p1_valid  : (K,2) ndarray
            movements : (K,)  ndarray
        """
        if len(pts_t0) == 0:
            return np.array([]), np.array([]), np.array([])

        lk_params = dict(
            winSize=(21, 21),       # 15→21: daha geniş pencere → büyük hareketleri yakala
            maxLevel=4,             # 3→4: bir piramit seviyesi daha → daha büyük yer değişimi
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # IMU-aided initial guess
        pts_init = self._compute_rotation_initial_guess(pts_t0)

        # ── Forward: t0 → t1 ─────────────────────────────────────
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            rect_t0, rect_t1, pts_t0, pts_init, **lk_params)

        # ── Backward: t1 → t0 ────────────────────────────────────
        p0_back, st0, _ = cv2.calcOpticalFlowPyrLK(
            rect_t1, rect_t0, p1, pts_t0, **lk_params)

        # ── Forward-backward konsistans kontrolü ─────────────────
        mask_fwd = st1.ravel() == 1
        mask_bwd = st0.ravel() == 1

        # İleri-geri mesafesi
        fb_dist = np.linalg.norm(
            pts_t0.reshape(-1, 2) - p0_back.reshape(-1, 2), axis=1)

        # Tüm koşulları birleştir: forward OK + backward OK + fb_dist < 1.0px
        mask = mask_fwd & mask_bwd & (fb_dist < 1.0)

        p0_valid = pts_t0[mask].reshape(-1, 2)
        p1_valid = p1[mask].reshape(-1, 2)
        movements = np.hypot(p1_valid[:, 0] - p0_valid[:, 0],
                             p1_valid[:, 1] - p0_valid[:, 1])

        return p0_valid, p1_valid, movements

    # ── 3D-2D eşleştirme (vektörize) ─────────────────────────────────────────
    def match_3d_2d(self, pts_t0:      np.ndarray,
                    pts_3d:      np.ndarray,
                    p0_arr:      np.ndarray,
                    p1_arr:      np.ndarray,
                    dist_thresh: float = 2.0):   # 1.5 → 2.0: biraz daha toleranslı
        """
        Optical flow sonucu gelen p0 noktalarını ORB keypoint'leriyle eşleştir.
        """
        if len(p0_arr) == 0 or len(pts_3d) == 0:
            return np.array([]), np.array([])

        pts_arr = pts_t0.reshape(-1, 2)

        diff  = p0_arr[:, None, :] - pts_arr[None, :, :]
        dists = np.linalg.norm(diff, axis=2)

        min_idx  = np.argmin(dists, axis=1)
        min_dist = dists[np.arange(len(p0_arr)), min_idx]

        valid = min_dist < dist_thresh
        if not np.any(valid):
            return np.array([]), np.array([])

        matched_3d = pts_3d[min_idx[valid]].astype(np.float32)
        matched_2d = p1_arr[valid].astype(np.float32)
        return matched_3d, matched_2d

    # ── PnP odometry (SQPNP + LM Refinement) ──────────────────────────────────
    def calculate_odometry(self, pts_3d: np.ndarray, pts_2d: np.ndarray):
        """
        SQPNP + RANSAC ile odometry, ardından inlier'larla LM refinement.

        Dönüş:
            rvec, tvec, inliers  veya  None, None, None
        """
        if len(pts_3d) < 10:
            return None, None, None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.reshape(-1, 1, 3),
            pts_2d.reshape(-1, 1, 2),
            self.K_rect,
            None,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount=500,
            flags=cv2.SOLVEPNP_SQPNP
        )

        if success and inliers is not None and len(inliers) >= 10:
            # ── LM Refinement: inlier noktalarla iteratif iyileştirme ────────
            try:
                inlier_idx = inliers.ravel()
                inlier_3d = pts_3d[inlier_idx].reshape(-1, 1, 3)
                inlier_2d = pts_2d[inlier_idx].reshape(-1, 1, 2)
                rvec, tvec = cv2.solvePnPRefineLM(
                    inlier_3d, inlier_2d,
                    self.K_rect, None, rvec, tvec
                )
            except cv2.error:
                pass  # Refinement başarısız → orijinal sonuçla devam
            return rvec, tvec, inliers
        return None, None, None

    # ── Kalman adımı: IMU predict + Pre-filter + PnP update ───────────────────
    def kalman_step(self, tvec: np.ndarray, dt: float, n_inliers: int) -> str:
        """
        1. Güncel Kalman bias tahminini IMU'ya yaz
        2. IMU ortalama ivmesiyle Kalman'ı öngör (predict)
        3. Pre-filter: absürt hız ölçümlerini Kalman'a vermeden reddet
        4. PnP tvec ile Kalman'ı güncelle (update) — innovation gate otomatik

        Dönüş:
            "updated"         — normal Kalman update yapıldı
            "innov_rejected"  — innovation gate reddetti
            "low_inliers"     — düşük inlier, predict-only
            "abs_thresh"      — mutlak hız eşiği aşıldı
            "ratio_thresh"    — oransal hız eşiği aşıldı
            "direction_thresh" — yön kontrolü reddetti
        """
        # Bias geri besleme: Kalman → IMU
        self.imu.set_bias(self.kalman.gyro_bias)

        # Ortalama dünya ivmesi → predict (sınırlandırılmış)
        accel_world = self.imu.get_accel_world_mean()
        accel_norm = np.linalg.norm(accel_world)
        if accel_norm > 5.0:
            accel_world = accel_world * (5.0 / accel_norm)
        self.kalman.predict(accel_world, dt)

        # Düşük inlier kontrolü: çok az inlier → update yapma
        if n_inliers < self.MIN_INLIERS_FOR_UPDATE:
            return "low_inliers"

        # ── PRE-FILTER: Absürt hız ölçümlerini Kalman'a vermeden reddet ──────
        v_meas = tvec.ravel() / max(dt, 1e-9)
        v_meas_speed = np.linalg.norm(v_meas)

        # 1) Mutlak hız eşiği — fiziksel olarak imkansız (kullanıcı: 4 m/s)
        MAX_VELOCITY_THRESH = 3.0   # m/s (GT max ~2 m/s, güvenlik payı)
        if v_meas_speed > MAX_VELOCITY_THRESH:
            return "abs_thresh"

        # 2) Kalman state'ine göre oransal eşik
        current_speed = np.linalg.norm(self.kalman.x[3:6])
        if current_speed > 0.1:
            speed_ratio = v_meas_speed / current_speed
            if speed_ratio > 3.5:   # 3.5 kattan fazla → absürt
                return "ratio_thresh"

        # 3) Yön kontrolü — 120°+ ani yön değişimi → PnP hatası
        if current_speed > 0.1 and v_meas_speed > 0.1:
            cos_angle = np.dot(v_meas, self.kalman.x[3:6]) / (
                v_meas_speed * current_speed)
            if cos_angle < -0.5:    # >120° → absürt
                return "direction_thresh"

        # ── PnP tvec → update (innovation gate Kalman içinde) ────────────────
        self.kalman.update(tvec.ravel(), dt, n_inliers)

        if not self.kalman.last_update_accepted:
            return "innov_rejected"
        return "updated"

    # ── Sadece predict step (PnP başarısız olduğunda) ─────────────────────────
    def kalman_predict_only(self, dt: float):
        """PnP başarısız olduğunda sadece predict ile devam. IMU integration yok."""
        self.kalman.predict_only(dt)
        self.imu.reset()

    # ── Hız sürekliliği kontrolü ──────────────────────────────────────────────
    def check_speed_continuity(self, speed: float, max_ratio: float = 3.0) -> bool:
        """
        Önceki frame hızı ile mevcut hız arasındaki oranı kontrol et.
        Çok ani değişim → muhtemelen hatalı.

        İlk birkaç frame'de (prev_speed ≈ 0) kontrolü geç.
        """
        if self._prev_speed < 0.05:  # Önceki hız çok düşük → sınırlama uygulama
            self._prev_speed = speed
            return True

        if speed > 0.05:
            ratio = speed / max(self._prev_speed, 0.01)
            if ratio > max_ratio:
                return False

        self._prev_speed = speed
        return True

    # ── Hız erişimi (EMA filtreli) ─────────────────────────────────────────────
    @property
    def speed_ms(self) -> float:
        return float(np.linalg.norm(self.velocity_ms))

    @property
    def velocity_ms(self) -> np.ndarray:
        raw = self.kalman.velocity
        if not self._ema_initialized:
            self._ema_velocity = raw.copy()
            self._ema_initialized = True
            return raw
        self._ema_velocity = (
            self._ema_alpha * raw +
            (1.0 - self._ema_alpha) * self._ema_velocity
        )
        return self._ema_velocity.copy()