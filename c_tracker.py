# “””
c_tracker.py  —  Stereo Visual-Inertial Odometry  (Drone / TUM-VI)

State vector  (9D):
x = [px, py, pz,   vx, vy, vz,   bgx, bgy, bgz]
pozisyon(m)   hız(m/s)       gyro bias (rad/s)

Kalman Predict  →  IMU kinematik modeli
Kalman Update   →  PnP tvec ölçümü (SQPNP)

Yazar: sıfırdan yazıldı  –  EPnP kaldırıldı, SQPNP eklendi
“””

import cv2
import numpy as np
from config import k_left, d_left, k_right, d_right, r_matrix, t_vector, baseline

# ─────────────────────────────────────────────────────────────────────────────

# 1.  9D KALMAN FİLTRESİ

# ─────────────────────────────────────────────────────────────────────────────

class KalmanVIO:
“””
State  : [px py pz  vx vy vz  bgx bgy bgz]  (9,)
Ölçüm  : [px py pz]  — PnP’den gelen delta pozisyon        (3,)

```
F (geçiş):
    px += vx*dt
    py += vy*dt
    pz += vz*dt
    vx += ax_imu * dt          (ax_imu = a_dünya)
    vy += ay_imu * dt
    vz += az_imu * dt
    bg* = sabit  (random walk)

Q  : süreç gürültüsü  (IMU noise)
R  : ölçüm gürültüsü  (PnP noise)
"""

# Gürültü parametreleri  — TUM-VI IMU datasheet'e yakın değerler
ACCEL_NOISE   = 0.01      # m/s² / √Hz
GYRO_NOISE    = 0.005     # rad/s / √Hz
BIAS_NOISE    = 1e-5      # rad/s²  (bias random walk)
PNP_POS_NOISE = 0.05      # m  (PnP ölçüm gürültüsü)

def __init__(self):
    self.x  = np.zeros(9)            # state
    self.P  = np.eye(9) * 0.1        # kovaryans
    self.gravity = np.array([0.0, 0.0, 9.81])
    self._initialized = False

def reset(self):
    self.x  = np.zeros(9)
    self.P  = np.eye(9) * 0.1
    self._initialized = False

# ------------------------------------------------------------------
def predict(self, gyro_raw, accel_raw, R_body2world, dt):
    """
    IMU ölçümü ile state öne taşı.

    gyro_raw      : (3,) ham gyro  [rad/s]
    accel_raw     : (3,) ham ivme  [m/s²]
    R_body2world  : (3,3) mevcut rotasyon tahmini (PnP'den)
    dt            : zaman adımı [s]
    """
    if dt <= 0:
        return

    bg = self.x[6:9]                          # gyro bias tahmini
    gyro_corrected = gyro_raw - bg            # bias çıkar

    # Dünya uzayında linear ivme (yerçekimi çıkar)
    a_world = R_body2world @ accel_raw - self.gravity

    # Durum geçiş matrisi F  (9×9)
    F = np.eye(9)
    F[0, 3] = dt;  F[1, 4] = dt;  F[2, 5] = dt   # p += v*dt

    # Süreç gürültüsü Q
    q_pos  = (0.5 * self.ACCEL_NOISE * dt**2) ** 2
    q_vel  = (self.ACCEL_NOISE * dt) ** 2
    q_bias = (self.BIAS_NOISE  * dt) ** 2
    Q = np.diag([q_pos]*3 + [q_vel]*3 + [q_bias]*3)

    # State güncelle
    self.x[0:3] += self.x[3:6] * dt + 0.5 * a_world * dt**2
    self.x[3:6] += a_world * dt
    # bias random walk — state değişmez, sadece kovaryans büyür

    # Kovaryans öne taşı
    self.P = F @ self.P @ F.T + Q

# ------------------------------------------------------------------
def update(self, delta_pos_pnp, n_inliers):
    """
    PnP'den gelen delta pozisyon ile Kalman güncelle.

    delta_pos_pnp : (3,) tvec (kamera frame → dünya frame dönüşümü
                     main'de yapılmalı, burada sadece (3,) beklenir)
    n_inliers     : PnP inlier sayısı  (ölçüm güvenilirliğini ölçekler)
    """
    H = np.zeros((3, 9))
    H[0:3, 0:3] = np.eye(3)          # ölçüm sadece pozisyonu görür

    # İnlier sayısı arttıkça ölçüme güven artar
    r_scale = max(0.2, 1.0 - n_inliers * 0.01)
    R_meas  = np.eye(3) * (self.PNP_POS_NOISE * r_scale) ** 2

    S = H @ self.P @ H.T + R_meas    # inovasyon kovaryansı
    K = self.P @ H.T @ np.linalg.inv(S)   # Kalman kazancı

    innovation   = delta_pos_pnp - H @ self.x
    self.x      += K @ innovation
    self.P       = (np.eye(9) - K @ H) @ self.P

    self._initialized = True

# ------------------------------------------------------------------
def get_velocity(self):
    """Filtrelenmiş hız vektörü (3,) m/s"""
    return self.x[3:6].copy()

def get_speed(self):
    """Skaler hız m/s"""
    return float(np.linalg.norm(self.x[3:6]))

def get_position(self):
    return self.x[0:3].copy()

def get_gyro_bias(self):
    return self.x[6:9].copy()
```

# ─────────────────────────────────────────────────────────────────────────────

# 2.  IMU PREINTEGRATOR  (Kalman predict için yardımcı)

# ─────────────────────────────────────────────────────────────────────────────

class IMUPreintegrator:
“””
İki frame arasındaki IMU örneklerini entegre eder.
Kalman predict’e ham örnek yerine,
ortalama gyro + accel + kümülatif ΔR sağlar.
“””

```
def __init__(self):
    self.gravity = np.array([0.0, 0.0, 9.81])
    self.reset()

def reset(self):
    self.delta_R   = np.eye(3)
    self.delta_v   = np.zeros(3)
    self.delta_p   = np.zeros(3)
    self.dt_total  = 0.0
    self.gyro_acc  = np.zeros(3)   # ortalama için biriktir
    self.accel_acc = np.zeros(3)
    self.n_samples = 0

def integrate(self, gyro, accel, dt):
    if dt <= 0:
        return
    # Rodrigues dönüşümü
    angle = np.linalg.norm(gyro) * dt
    if angle > 1e-10:
        axis = gyro / np.linalg.norm(gyro)
        K = np.array([
            [ 0,       -axis[2],  axis[1]],
            [ axis[2],  0,       -axis[0]],
            [-axis[1],  axis[0],  0      ]
        ])
        dR = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
    else:
        dR = np.eye(3)

    a_world        = self.delta_R @ accel - self.gravity
    self.delta_p  += self.delta_v * dt + 0.5 * a_world * dt**2
    self.delta_v  += a_world * dt
    self.delta_R   = self.delta_R @ dR
    self.dt_total += dt

    self.gyro_acc  += gyro * dt
    self.accel_acc += accel * dt
    self.n_samples += 1

def get_prediction(self):
    return self.delta_R.copy(), self.delta_v.copy(), self.delta_p.copy()

def get_mean_imu(self):
    """Kalman predict için ağırlıklı ortalama gyro/accel döndür"""
    if self.dt_total > 0:
        return (self.gyro_acc  / self.dt_total,
                self.accel_acc / self.dt_total)
    return np.zeros(3), np.zeros(3)
```

# ─────────────────────────────────────────────────────────────────────────────

# 3.  STEREO ODOMETRY TRACKER

# ─────────────────────────────────────────────────────────────────────────────

class StereoOdometryTracker:

```
# Güvenli derinlik aralığı (metre) — TUM-VI iç mekân
Z_MIN, Z_MAX = 0.3, 15.0
# LK flow başarısız sayılan max piksel mesafesi
FLOW_MAX_DIST = 150.0
# 3D-2D eşleştirme piksel toleransı
MATCH_TOL = 2.0
# Hız sınırı filtresi (m/s)  — drone için 12 m/s
SPEED_LIMIT = 12.0

def __init__(self):
    # ORB dedektörü
    self.orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=10
    )

    # Kalman + IMU
    self.kalman = KalmanVIO()
    self.imu    = IMUPreintegrator()

    # Rotasyon tahmini (PnP'den güncellenir)
    self._R_body2world = np.eye(3)

    # Stereo rektifikasyon (fisheye)
    result = cv2.fisheye.stereoRectify(
        k_left, d_left, k_right, d_right,
        (512, 512), r_matrix, t_vector,
        flags=cv2.CALIB_ZERO_DISPARITY
    )
    self.R1, self.R2, self.P1, self.P2, self.Q = result[:5]

    self.f_ideal = float(self.P1[0, 0])
    self.K_rect  = self.P1[:3, :3].copy()

    # SGBM — TUM-VI için optimize
    self.sgbm = cv2.StereoSGBM_create(
        minDisparity    = 0,
        numDisparities  = 128,       # 64→128: yakın nesneler için
        blockSize       = 9,         # 11→9: daha az bulanıklık
        P1              = 8  * 3 * 9**2,
        P2              = 32 * 3 * 9**2,
        disp12MaxDiff   = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange    = 32,
        mode            = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    self._maps_ready = False

# ------------------------------------------------------------------
def _ensure_maps(self, h, w):
    if self._maps_ready:
        return
    self._ml1, self._ml2 = cv2.fisheye.initUndistortRectifyMap(
        k_left,  d_left,  self.R1, self.P1, (w, h), cv2.CV_32FC1)
    self._mr1, self._mr2 = cv2.fisheye.initUndistortRectifyMap(
        k_right, d_right, self.R2, self.P2, (w, h), cv2.CV_32FC1)
    self._maps_ready = True

# ------------------------------------------------------------------
def rectify_image(self, img, side='left'):
    h, w = img.shape[:2]
    self._ensure_maps(h, w)
    maps = (self._ml1, self._ml2) if side == 'left' else (self._mr1, self._mr2)
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)

# ------------------------------------------------------------------
def process_space_get_depth(self, raw_left, raw_right):
    """
    Döndürür:
      pts_rect  (N,1,2) float32  — rectified sol pikseller
      pts_3d    (N,3)   float32  — 3D noktalar metre
      rect_l            — rectified sol görüntü (flow için)
    """
    h, w = raw_left.shape[:2]
    self._ensure_maps(h, w)

    rect_l = cv2.remap(raw_left,  self._ml1, self._ml2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(raw_right, self._mr1, self._mr2, cv2.INTER_LINEAR)

    disp = self.sgbm.compute(rect_l, rect_r).astype(np.float32) / 16.0

    kp_list, _ = self.orb.detectAndCompute(rect_l, None)
    if not kp_list:
        return np.empty((0,1,2), np.float32), np.empty((0,3), np.float32), rect_l

    cx, cy = float(self.P1[0, 2]), float(self.P1[1, 2])
    fx, fy = float(self.P1[0, 0]), float(self.P1[1, 1])

    # Vektörize hesaplama (for döngüsü yok)
    pts = np.array([kp.pt for kp in kp_list], dtype=np.float32)  # (N,2)
    xi  = np.round(pts[:, 0]).astype(int)
    yi  = np.round(pts[:, 1]).astype(int)

    # Sınır maskesi
    in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    pts = pts[in_bounds];  xi = xi[in_bounds];  yi = yi[in_bounds]

    d_vals = disp[yi, xi]                      # disparity değerleri
    valid  = d_vals >= 0.5
    pts    = pts[valid];  d_vals = d_vals[valid]
    xi     = xi[valid];   yi     = yi[valid]

    z_vals = (self.f_ideal * baseline) / d_vals
    depth_ok = (z_vals > self.Z_MIN) & (z_vals < self.Z_MAX)
    pts    = pts[depth_ok];  z_vals = z_vals[depth_ok]

    if len(pts) == 0:
        return np.empty((0,1,2), np.float32), np.empty((0,3), np.float32), rect_l

    X = (pts[:, 0] - cx) * z_vals / fx
    Y = (pts[:, 1] - cy) * z_vals / fy

    pts_rect = pts.reshape(-1, 1, 2)
    pts_3d   = np.stack([X, Y, z_vals], axis=1).astype(np.float32)
    return pts_rect, pts_3d, rect_l

# ------------------------------------------------------------------
def track_time_get_flow(self, rect_t0, rect_t1, pts_t0):
    """
    Lucas-Kanade optical flow.
    Döndürür: (p0_arr, p1_arr, movements) — hepsi numpy array
    """
    if len(pts_t0) == 0:
        return (np.empty((0,2), np.float32),
                np.empty((0,2), np.float32),
                np.empty(0,     np.float32))

    lk_params = dict(
        winSize  = (15, 15),          # 21→15: daha hızlı
        maxLevel = 3,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        rect_t0, rect_t1, pts_t0, None, **lk_params)

    # Geri-akış doğrulaması (forward-backward check)
    p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
        rect_t1, rect_t0, p1, None, **lk_params)

    fb_err  = np.abs(pts_t0 - p0_back).reshape(-1, 2).max(axis=1)
    mask    = (st.ravel() == 1) & (st_back.ravel() == 1) & (fb_err < 1.0)

    p0_valid = pts_t0.reshape(-1, 2)[mask]
    p1_valid = p1.reshape(-1, 2)[mask]

    if len(p0_valid) == 0:
        return (np.empty((0,2), np.float32),
                np.empty((0,2), np.float32),
                np.empty(0,     np.float32))

    movements = np.hypot(
        p1_valid[:, 0] - p0_valid[:, 0],
        p1_valid[:, 1] - p0_valid[:, 1]
    )

    # Aşırı hareket eden noktaları at (outlier)
    ok = movements < self.FLOW_MAX_DIST
    return p0_valid[ok], p1_valid[ok], movements[ok]

# ------------------------------------------------------------------
def match_3d_2d(self, pts_t0_rect, pts_3d_t0, p0_flow, p1_flow):
    """
    Optical flow'dan gelen p0 noktalarını pts_t0 ile eşleştir.
    Vektörize KD-tree benzeri numpy broadcasting kullanır.

    Döndürür:
      matched_3d  (M,3)
      matched_2d  (M,2)
    """
    if len(p0_flow) == 0 or len(pts_t0_rect) == 0:
        return np.empty((0,3), np.float32), np.empty((0,2), np.float32)

    base = pts_t0_rect.reshape(-1, 2)         # (N,2)
    query = p0_flow                            # (M,2)

    # Broadcasting: (M,1,2) - (1,N,2) → (M,N,2) → (M,N) mesafe
    diff  = query[:, np.newaxis, :] - base[np.newaxis, :, :]   # (M,N,2)
    dists = np.linalg.norm(diff, axis=2)                        # (M,N)
    idx   = np.argmin(dists, axis=1)                            # (M,)
    min_d = dists[np.arange(len(query)), idx]                   # (M,)

    ok = min_d < self.MATCH_TOL
    matched_3d = pts_3d_t0[idx[ok]]
    matched_2d = p1_flow[ok]
    return matched_3d, matched_2d

# ------------------------------------------------------------------
def calculate_odometry(self, pts_3d, pts_2d):
    """
    SQPNP ile kamera hareketi tahmini.
    Döndürür: (rvec, tvec, inliers)  —  başarısızda (None, None, None)
    """
    if len(pts_3d) < 6:
        return None, None, None

    obj = pts_3d.reshape(-1, 1, 3).astype(np.float32)
    img = pts_2d.reshape(-1, 1, 2).astype(np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img,
        self.K_rect,
        None,                          # distortion yok (rect uzay)
        reprojectionError = 1.5,       # 2.0→1.5: daha katı
        confidence        = 0.99,
        iterationsCount   = 300,
        flags             = cv2.SOLVEPNP_SQPNP   # EPnP → SQPNP
    )

    if not success or inliers is None or len(inliers) < 6:
        return None, None, None

    # Rotasyon matrisini güncelle (Kalman predict için)
    R_delta, _ = cv2.Rodrigues(rvec)
    self._R_body2world = R_delta @ self._R_body2world

    return rvec, tvec, inliers

# ------------------------------------------------------------------
def kalman_predict_from_imu(self, dt):
    """
    IMU preintegrator'ın biriktirdiği ortalama ölçümü
    Kalman predict adımına besle.
    """
    mean_gyro, mean_accel = self.imu.get_mean_imu()
    self.kalman.predict(mean_gyro, mean_accel, self._R_body2world, dt)

# ------------------------------------------------------------------
def kalman_update_from_pnp(self, tvec, n_inliers):
    """
    PnP tvec'ini (3,) alarak Kalman update adımını çalıştır.
    Döndürür: filtrelenmiş hız (m/s skaler)
    """
    dp = tvec.ravel()[:3].astype(np.float64)
    self.kalman.update(dp, n_inliers)
    return self.kalman.get_speed()
```