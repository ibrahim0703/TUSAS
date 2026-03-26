"""
tracker_eskf.py  —  Stereo Visual-Inertial Odometry (V2: 15D ESKF)
====================================================================
15D Error-State Kalman Filter ile drone hiz tahmini.
TUM-VI Room1 dataseti icin optimize edilmis.

Nominal State (16D):
    [p(3), v(3), q(4), ba(3), bg(3)]

Error State (15D):
    [dp(3), dv(3), dtheta(3), dba(3), dbg(3)]

Predict -> Per-IMU-sample quaternion-based propagation
Update  -> Direct 6DOF Pose measurement (Asama 2: pozisyon + oryantasyon)
Tracking -> Continuous feature tracking (Asama 3: track_id + sliding window)
"""

import cv2
import numpy as np
from config import k_left, d_left, k_right, d_right, r_matrix, t_vector, baseline


# =============================================================================
# QUATERNION UTILITIES
# =============================================================================

def skew(v):
    """3-vector -> 3x3 skew-symmetric matrix."""
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1], v[0],  0.0]
    ])


def quat_normalize(q):
    """Normalize quaternion to unit length. q = [w, x, y, z]."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mult(q1, q2):
    """Hamilton quaternion multiplication. q = [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conj(q):
    """Quaternion conjugate. q = [w, x, y, z] -> [w, -x, -y, -z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotmat(q):
    """Unit quaternion -> 3x3 rotation matrix (body -> world)."""
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])


def rotmat_to_quat(R):
    """3x3 rotation matrix -> unit quaternion [w, x, y, z]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def quat_from_axis_angle(aa):
    """Axis-angle vector (3D) -> quaternion [w, x, y, z]."""
    angle = np.linalg.norm(aa)
    if angle < 1e-12:
        return quat_normalize(np.array([1.0, aa[0]*0.5, aa[1]*0.5, aa[2]*0.5]))
    axis = aa / angle
    ha = angle * 0.5
    sa = np.sin(ha)
    return np.array([np.cos(ha), axis[0]*sa, axis[1]*sa, axis[2]*sa])


# =============================================================================
# 15D ERROR-STATE KALMAN FILTER
# =============================================================================

class ESKF15:
    """
    15-dimensional Error-State Kalman Filter.

    Nominal State (16D): [p(3), v(3), q(4), ba(3), bg(3)]
    Error State (15D):   [dp(3), dv(3), dtheta(3), dba(3), dbg(3)]

    Asama 2: Update sadece 6DOF — pozisyon + oryantasyon.
    Hiz, P matrisindeki capraz korelasyonlar uzerinden dolayli duzeltilir.
    """

    def __init__(self,
                 sigma_accel:      float = 0.0396,
                 sigma_gyro:       float = 0.00226,
                 sigma_bias_gyro:  float = 0.00000156,
                 sigma_bias_accel: float = 0.01,
                 sigma_pos:        float = 0.05,
                 sigma_vel:        float = 0.3,
                 sigma_orient:     float = 0.03,
                 innov_gate:       float = 22.0):

        self.n = 15  # error state dimension

        # -- Nominal state --------------------------------------------------
        self.p  = np.zeros(3)
        self.v  = np.zeros(3)
        self.q  = np.array([1.0, 0.0, 0.0, 0.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)

        # -- Error state covariance (15x15) ---------------------------------
        self.P = np.zeros((self.n, self.n))
        self.P[0:3, 0:3]    = np.eye(3) * 0.1
        self.P[3:6, 3:6]    = np.eye(3) * 0.1
        self.P[6:9, 6:9]    = np.eye(3) * 0.01
        self.P[9:12, 9:12]  = np.eye(3) * 0.01
        self.P[12:15, 12:15] = np.eye(3) * 1e-4

        # -- Noise parameters -----------------------------------------------
        self.sa  = sigma_accel
        self.sg  = sigma_gyro
        self.sba = sigma_bias_accel
        self.sbg = sigma_bias_gyro
        self.sp  = sigma_pos
        self.sv  = sigma_vel
        self.so  = sigma_orient

        # -- Gravity (world frame, z-up) ------------------------------------
        self.gravity = np.array([0.0, 0.0, 9.81])

        # -- Innovation gate ------------------------------------------------
        self.innov_gate = innov_gate
        self._initialized = False
        self._last_update_accepted = True
        self._consecutive_rejections = 0
        self._MAX_CONSECUTIVE_REJECT = 3

    # -- Initialization -----------------------------------------------------

    def initialize_from_gravity(self, accel_mean: np.ndarray):
        """
        Ilk IMU ivme verilerinden oryantasyonu hesapla (gravity alignment).
        """
        a = accel_mean.ravel()
        a_norm = np.linalg.norm(a)
        if a_norm < 1.0:
            self._initialized = True
            return

        g_body = a / a_norm
        g_world = np.array([0.0, 0.0, 1.0])

        cross = np.cross(g_body, g_world)
        dot = float(np.dot(g_body, g_world))

        if dot > 0.9999:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])
        elif dot < -0.9999:
            if abs(g_body[0]) < 0.9:
                perp = np.cross(g_body, np.array([1.0, 0.0, 0.0]))
            else:
                perp = np.cross(g_body, np.array([0.0, 1.0, 0.0]))
            perp = perp / np.linalg.norm(perp)
            self.q = np.array([0.0, perp[0], perp[1], perp[2]])
        else:
            s = np.sqrt(2.0 * (1.0 + dot))
            self.q = quat_normalize(np.array([
                s / 2.0, cross[0] / s, cross[1] / s, cross[2] / s
            ]))

        self._initialized = True

    def initialize(self, v0: np.ndarray):
        """Ilk hiz olcumuyle baslat."""
        self.v = v0.ravel().copy()
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def last_update_accepted(self) -> bool:
        return self._last_update_accepted

    # -- PREDICT: Per-IMU-sample propagation --------------------------------

    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        """
        IMU-driven predict. Her IMU ornegi icin cagirilir.
        Nominal state propagation + error covariance propagation.
        """
        if not self._initialized:
            return

        omega  = gyro.ravel() - self.bg
        a_body = accel.ravel() - self.ba

        R = quat_to_rotmat(self.q)
        a_world = R @ a_body - self.gravity

        a_norm = np.linalg.norm(a_world)
        if a_norm > 20.0:
            a_world = a_world * (20.0 / a_norm)

        # == Nominal state propagation =====================================
        dt2 = dt * dt
        self.p = self.p + self.v * dt + 0.5 * a_world * dt2
        self.v = self.v + a_world * dt
        self.q = quat_normalize(quat_mult(self.q, quat_from_axis_angle(omega * dt)))

        # == Error state transition F (15x15) ==============================
        F = np.eye(self.n)
        F[0:3, 3:6]  = np.eye(3) * dt
        F[3:6, 6:9]  = -R @ skew(a_body) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 6:9]   = np.eye(3) - skew(omega) * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        # == Process noise Q (15x15) =======================================
        Q = np.zeros((self.n, self.n))
        Q[3:6, 3:6]     = np.eye(3) * (self.sa ** 2 * dt)
        Q[6:9, 6:9]     = np.eye(3) * (self.sg ** 2 * dt)
        Q[9:12, 9:12]   = np.eye(3) * (self.sba ** 2 * dt)
        Q[12:15, 12:15] = np.eye(3) * (self.sbg ** 2 * dt)

        # == Covariance propagation ========================================
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

        floor_vals = np.array(
            [1e-6]*3 + [1e-6]*3 + [1e-8]*3 + [1e-8]*3 + [1e-8]*3
        )
        diag = np.diag(self.P)
        np.fill_diagonal(self.P, np.maximum(diag, floor_vals))

    # -- UPDATE: 9DOF Direct Pose + Velocity (Asama 2 rev) -------------------

    def update_pose(self, p_meas: np.ndarray, q_meas: np.ndarray,
                    n_inliers: int, v_pseudo: np.ndarray = None):
        """
        9DOF pose update — pozisyon + hiz (pseudo) + oryantasyon.

        H matrisi (9x15):
            [I  0  0  0  0]   dp -> pozisyon
            [0  I  0  0  0]   dv -> hiz
            [0  0  I  0  0]   dtheta -> oryantasyon

        v_pseudo = None ise 6DOF'a duser (geriye uyumlu).

        Args:
            p_meas   : World frame pozisyon olcumu (3,)
            q_meas   : World frame quaternion olcumu [w,x,y,z] (4,)
            n_inliers: PnP inlier sayisi
            v_pseudo : World frame hiz pseudo-olcumu (3,) veya None
        """
        if not self._initialized:
            self.p = p_meas.ravel().copy()
            self.q = quat_normalize(q_meas.ravel().copy())
            if v_pseudo is not None:
                self.v = v_pseudo.ravel().copy()
            self._initialized = True
            return

        # -- Adaptive measurement noise ------------------------------------
        inlier_factor = max(1.0, 50.0 / max(n_inliers, 1))

        use_vel = v_pseudo is not None
        m_dim = 9 if use_vel else 6

        R_noise = np.zeros((m_dim, m_dim))

        if use_vel:
            # 9DOF: position + velocity + orientation
            R_noise[0:3, 0:3] = np.eye(3) * (self.sp ** 2) * inlier_factor
            R_noise[3:6, 3:6] = np.eye(3) * (self.sv ** 2) * inlier_factor
            R_noise[6:9, 6:9] = np.eye(3) * (self.so ** 2) * inlier_factor
        else:
            # 6DOF: position + orientation (fallback)
            R_noise[0:3, 0:3] = np.eye(3) * (self.sp ** 2) * inlier_factor
            R_noise[3:6, 3:6] = np.eye(3) * (self.so ** 2) * inlier_factor

        # -- Innovation ----------------------------------------------------
        y_p = p_meas.ravel() - self.p

        dq = quat_mult(q_meas.ravel(), quat_conj(self.q))
        if dq[0] < 0:
            dq = -dq
        y_theta = 2.0 * dq[1:4]

        if use_vel:
            y_v = v_pseudo.ravel() - self.v
            y = np.concatenate([y_p, y_v, y_theta])  # (9,)
        else:
            y = np.concatenate([y_p, y_theta])  # (6,)

        # -- H matrix ------------------------------------------------------
        H = np.zeros((m_dim, self.n))
        if use_vel:
            H[0:3, 0:3] = np.eye(3)   # dp -> position
            H[3:6, 3:6] = np.eye(3)   # dv -> velocity
            H[6:9, 6:9] = np.eye(3)   # dtheta -> orientation
        else:
            H[0:3, 0:3] = np.eye(3)   # dp -> position
            H[3:6, 6:9] = np.eye(3)   # dtheta -> orientation

        # -- Innovation gate (chi-sq) --------------------------------------
        S = H @ self.P @ H.T + R_noise
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self._last_update_accepted = False
            return

        mahal_sq = float(y.T @ S_inv @ y)

        if mahal_sq > self.innov_gate:
            self._consecutive_rejections += 1
            inflate = 0.1 * self._consecutive_rejections
            self.P[3:6, 3:6] += np.eye(3) * inflate
            if self._consecutive_rejections < self._MAX_CONSECUTIVE_REJECT:
                self._last_update_accepted = False
                return

        self._last_update_accepted = True
        self._consecutive_rejections = 0

        # -- Kalman gain ---------------------------------------------------
        K = self.P @ H.T @ S_inv   # (15 x m_dim)

        # -- Error state correction ----------------------------------------
        dx = K @ y  # (15,)

        # -- Inject error into nominal state -------------------------------
        self.p  += dx[0:3]
        self.v  += dx[3:6]
        dtheta = dx[6:9]
        dq_corr = quat_from_axis_angle(dtheta)
        self.q = quat_normalize(quat_mult(self.q, dq_corr))
        self.ba += dx[9:12]
        self.bg += dx[12:15]

        # -- Covariance update (Joseph form) -------------------------------
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_noise @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # -- Predict-only (no IMU data available) -------------------------------

    def predict_no_imu(self, dt: float):
        """IMU verisi yokken sabit hiz modeli ile predict."""
        if not self._initialized:
            return

        self.p = self.p + self.v * dt

        Q = np.zeros((self.n, self.n))
        Q[0:3, 0:3] = np.eye(3) * 0.01 * dt
        Q[3:6, 3:6] = np.eye(3) * 0.1 * dt
        Q[6:9, 6:9] = np.eye(3) * 0.01 * dt
        self.P += Q
        self.P = 0.5 * (self.P + self.P.T)

        self.v *= 0.998

    # -- Properties ---------------------------------------------------------

    @property
    def velocity(self) -> np.ndarray:
        return self.v.copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v))

    @property
    def position(self) -> np.ndarray:
        return self.p.copy()

    @property
    def quaternion(self) -> np.ndarray:
        return self.q.copy()

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quat_to_rotmat(self.q)

    @property
    def gyro_bias(self) -> np.ndarray:
        return self.bg.copy()

    @property
    def accel_bias(self) -> np.ndarray:
        return self.ba.copy()


# =============================================================================
# IMU PREINTEGRATOR (Sadece Optical Flow icin Delta-R)
# =============================================================================

class IMUPreintegrator:
    """
    IMU jiroskop verisinden kare ciftleri arasinda delta rotasyon biriktirir.
    Sadece optical flow initial guess (LK) icin kullanilir.
    """

    def __init__(self):
        self.bias_gyro = np.zeros(3)
        self.reset()

    def set_bias(self, bg: np.ndarray):
        self.bias_gyro = bg.ravel().copy()

    def reset(self):
        self.delta_R  = np.eye(3)
        self.dt_total = 0.0

    def integrate(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        """Jiroskop ile delta rotasyon biriktir (Rodrigues formulu)."""
        gyro_c = gyro - self.bias_gyro

        angle = np.linalg.norm(gyro_c) * dt
        if angle > 1e-10:
            axis = gyro_c / np.linalg.norm(gyro_c)
            K = skew(axis)
            dR = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        else:
            dR = np.eye(3)

        self.delta_R  = self.delta_R @ dR
        self.dt_total += dt

    def get_prediction(self):
        return self.delta_R.copy(), np.zeros(3), np.zeros(3)


# =============================================================================
# STEREO ODOMETRY TRACKER (V2: ESKF15 + Direct Pose + Continuous Tracking)
# =============================================================================

class StereoOdometryTracker:

    MIN_INLIERS_FOR_UPDATE = 20
    MIN_TRACKED_FEATURES = 150   # Asama 3: bu esik altinda yeni feature detect

    def __init__(self):
        # -- ORB dedector ---------------------------------------------------
        self.orb = cv2.ORB_create(
            nfeatures=1500,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=8
        )

        # -- GFTT parametreleri ---------------------------------------------
        self.gftt_params = dict(
            maxCorners=800,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )

        # -- IMU & ESKF ----------------------------------------------------
        self.imu    = IMUPreintegrator()
        self.kalman = ESKF15(
            sigma_accel=0.0396,
            sigma_gyro=0.00226,
            sigma_bias_gyro=0.00000156,
            sigma_bias_accel=0.0001,
            sigma_pos=0.05,
            sigma_orient=0.03,
            innov_gate=28.0
        )

        # -- Stereo rectification matrices ---------------------------------
        result = cv2.fisheye.stereoRectify(
            k_left, d_left, k_right, d_right,
            (512, 512), r_matrix, t_vector,
            flags=cv2.CALIB_ZERO_DISPARITY
        )
        self.R1, self.R2, self.P1, self.P2, self.Q = result[:5]

        self.f_ideal = self.P1[0, 0]
        self.K_rect  = self.P1[:3, :3]

        # -- Camera-IMU Extrinsic (TUM-VI standard calibration) ------------
        # T_cam0_imu: IMU frame -> cam0 (left camera) frame
        # p_cam = R_cam_imu @ p_imu + t_cam_imu
        R_cam_imu = np.array([
            [-0.9995250378696743,  0.029615343885863205, -0.008522328211654736],
            [ 0.0075019185074052044, -0.03439736061393144, -0.9993800792498829],
            [-0.02989013031643309, -0.998969345370175,     0.03415885127385616]
        ])
        t_cam_imu = np.array([0.04727988224914392, -0.047443232143367084, -0.0681999605066297])

        # Camera-to-body (IMU) rotation: R_imu_cam = R_cam_imu^T
        self._R_cam_to_body = R_cam_imu.T

        # Rectified camera frame -> body frame:
        #   rectified -> original camera (R1^T) -> body (R_cam_to_body)
        self._R_rect_to_body = self._R_cam_to_body @ self.R1.T

        # Camera position in body frame (lever arm)
        self._t_cam_in_body = -R_cam_imu.T @ t_cam_imu

        # -- SGBM stereo matcher -------------------------------------------
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=9,
            P1=8  * 3 * 9 ** 2,
            P2=32 * 3 * 9 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=5,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self._maps_ready = False

        # -- Asama 3: Continuous Tracking State ----------------------------
        self._prev_rect_L   = None           # onceki frame rectified sol goruntu
        self._tracked_pts_2d = None          # Nx2 float32 — onceki frame'deki 2D konumlar
        self._tracked_pts_3d = None          # Nx3 float32 — onceki frame'deki 3D noktalar
        self._track_ids      = np.array([], dtype=np.int64)
        self._next_track_id  = 0

        # -- Hiz kontrol ---------------------------------------------------
        self._prev_speed = 0.0
        self._ema_velocity = np.zeros(3)
        self._ema_initialized = False
        self._ema_alpha = 0.6

    # -- Rectification maps (lazy init) ------------------------------------

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

    # -- Stereo Rectification + Disparity ----------------------------------

    def rectify_stereo(self, raw_left: np.ndarray, raw_right: np.ndarray):
        """Rectify both images and compute SGBM disparity map."""
        h, w = raw_left.shape[:2]
        self._ensure_maps(h, w)
        rect_l = cv2.remap(raw_left,  self._ml1, self._ml2, cv2.INTER_LINEAR)
        rect_r = cv2.remap(raw_right, self._mr1, self._mr2, cv2.INTER_LINEAR)
        disp = self.sgbm.compute(rect_l, rect_r).astype(np.float32) / 16.0
        return rect_l, rect_r, disp

    # -- Hybrid Feature Detection: GFTT + ORB ------------------------------

    def _detect_features_hybrid(self, rect_img: np.ndarray):
        kp_orb, _ = self.orb.detectAndCompute(rect_img, None)
        pts_orb = np.array([kp.pt for kp in kp_orb], dtype=np.float32) \
            if kp_orb else np.empty((0, 2), dtype=np.float32)

        gftt = cv2.goodFeaturesToTrack(rect_img, **self.gftt_params)
        pts_gftt = gftt.reshape(-1, 2).astype(np.float32) \
            if gftt is not None else np.empty((0, 2), dtype=np.float32)

        if len(pts_orb) == 0 and len(pts_gftt) == 0:
            return np.empty((0, 2), dtype=np.float32)
        if len(pts_orb) == 0:
            return pts_gftt
        if len(pts_gftt) == 0:
            return pts_orb

        all_pts = np.vstack([pts_orb, pts_gftt])
        return self._deduplicate_points(all_pts, min_dist=3.0)

    @staticmethod
    def _deduplicate_points(pts: np.ndarray, min_dist: float = 3.0) -> np.ndarray:
        if len(pts) <= 1:
            return pts

        keep = np.ones(len(pts), dtype=bool)
        cell_size = min_dist
        cells = {}

        for i in range(len(pts)):
            if not keep[i]:
                continue
            cx, cy = int(pts[i, 0] / cell_size), int(pts[i, 1] / cell_size)
            found_neighbor = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (cx + dx, cy + dy)
                    if key in cells:
                        for j in cells[key]:
                            dist = np.hypot(pts[i, 0] - pts[j, 0],
                                            pts[i, 1] - pts[j, 1])
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

    # -- 2D Points -> 3D via Disparity Map ---------------------------------

    def _triangulate_points(self, pts_2d: np.ndarray,
                            disp: np.ndarray):
        """
        Nx2 piksel koordinatlari + disparity map -> Nx3 kamera frame 3D noktalar.
        Returns: (pts_3d, valid_mask)
        """
        h, w = disp.shape[:2]
        cx, cy = self.P1[0, 2], self.P1[1, 2]
        fx, fy = self.P1[0, 0], self.P1[1, 1]

        pts = pts_2d.reshape(-1, 2)
        n = len(pts)
        if n == 0:
            return np.empty((0, 3), dtype=np.float32), np.array([], dtype=bool)

        xi = np.round(pts[:, 0]).astype(int)
        yi = np.round(pts[:, 1]).astype(int)

        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

        d = np.zeros(n, dtype=np.float32)
        d[valid] = disp[yi[valid], xi[valid]]
        valid &= (d >= 1.0)

        z = np.zeros(n, dtype=np.float32)
        z[valid] = (self.f_ideal * baseline) / d[valid]
        valid &= (z > 0.3) & (z < 15.0)

        X = (pts[:, 0] - cx) * z / fx
        Y = (pts[:, 1] - cy) * z / fy
        pts_3d = np.stack([X, Y, z], axis=1).astype(np.float32)

        return pts_3d, valid

    # -- IMU-Aided Rotation Compensation -----------------------------------

    def _compute_rotation_initial_guess(self, pts_t0: np.ndarray) -> np.ndarray:
        delta_R = self.imu.delta_R

        angle = np.arccos(np.clip((np.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0))
        if angle < 0.001:
            return pts_t0.reshape(-1, 1, 2).astype(np.float32)

        pts = pts_t0.reshape(-1, 2)
        cx = self.P1[0, 2]
        cy = self.P1[1, 2]
        fx = self.P1[0, 0]
        fy = self.P1[1, 1]

        x_norm = (pts[:, 0] - cx) / fx
        y_norm = (pts[:, 1] - cy) / fy
        ones   = np.ones(len(pts))
        rays   = np.stack([x_norm, y_norm, ones], axis=1)

        rays_rotated = (delta_R @ rays.T).T

        z_r = rays_rotated[:, 2]
        z_r = np.where(np.abs(z_r) < 1e-6, 1e-6, z_r)
        x_pred = fx * (rays_rotated[:, 0] / z_r) + cx
        y_pred = fy * (rays_rotated[:, 1] / z_r) + cy

        pts_pred = np.stack([x_pred, y_pred], axis=1).astype(np.float32)
        return pts_pred.reshape(-1, 1, 2)

    # -- Asama 3: Continuous Feature Tracking --------------------------------

    @staticmethod
    def _filter_near_existing(new_pts: np.ndarray, existing_pts: np.ndarray,
                              min_dist: float = 8.0) -> np.ndarray:
        """Mevcut takip edilen noktalara yakin olan yeni noktalari cikar."""
        if len(existing_pts) == 0 or len(new_pts) == 0:
            return new_pts

        new_pts = new_pts.reshape(-1, 2)
        existing = existing_pts.reshape(-1, 2)

        keep = np.ones(len(new_pts), dtype=bool)
        batch = 500
        for start in range(0, len(new_pts), batch):
            end = min(start + batch, len(new_pts))
            diff = new_pts[start:end, None, :] - existing[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_dists = dists.min(axis=1)
            keep[start:end] = min_dists >= min_dist

        return new_pts[keep]

    def track_features(self, rect_L_new: np.ndarray):
        """
        Asama 3: Onceki frame'den yeni frame'e feature tracking (optical flow).

        Forward-backward kontrol ile kaliteli eslesmeler secilir.
        IMU-aided initial guess kullanilir.

        Returns:
            tracked_3d   (Mx3) — onceki frame kamera frame'inde 3D noktalar
            tracked_2d   (Mx2) — yeni frame'deki 2D konumlar
            survived_ids (M,)  — track ID'leri
        """
        if (self._prev_rect_L is None or
                self._tracked_pts_2d is None or
                len(self._tracked_pts_2d) == 0):
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 2), dtype=np.float32),
                    np.array([], dtype=np.int64))

        pts_prev = self._tracked_pts_2d.reshape(-1, 1, 2).astype(np.float32)

        # IMU-aided initial guess
        pts_init = self._compute_rotation_initial_guess(pts_prev)

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_rect_L, rect_L_new, pts_prev, pts_init, **lk_params)
        p0_back, st0, _ = cv2.calcOpticalFlowPyrLK(
            rect_L_new, self._prev_rect_L, p1, pts_prev, **lk_params)

        mask_fwd = st1.ravel() == 1
        mask_bwd = st0.ravel() == 1
        fb_dist = np.linalg.norm(
            pts_prev.reshape(-1, 2) - p0_back.reshape(-1, 2), axis=1)

        mask = mask_fwd & mask_bwd & (fb_dist < 1.0)

        # Bounds check
        h, w = rect_L_new.shape[:2]
        p1_flat = p1.reshape(-1, 2)
        mask &= ((p1_flat[:, 0] >= 0) & (p1_flat[:, 0] < w) &
                 (p1_flat[:, 1] >= 0) & (p1_flat[:, 1] < h))

        tracked_3d   = self._tracked_pts_3d[mask].copy()
        tracked_2d   = p1_flat[mask].copy()
        survived_ids = self._track_ids[mask].copy()

        return tracked_3d, tracked_2d, survived_ids

    def update_tracked_state(self, pts_2d: np.ndarray,
                             track_ids: np.ndarray,
                             rect_L: np.ndarray,
                             disp: np.ndarray):
        """
        Asama 3: Takip durumunu guncelle.

        1. Mevcut noktalar icin current stereo'dan yeniden 3D hesapla
        2. Takip sayisi esik altindaysa yeni GFTT+ORB feature'lar detect et
        3. Sonraki frame icin state'i sakla
        """
        pts_2d = pts_2d.reshape(-1, 2).astype(np.float32) \
            if len(pts_2d) > 0 else np.empty((0, 2), dtype=np.float32)

        if len(track_ids) == 0:
            track_ids = np.array([], dtype=np.int64)

        # Re-triangulate from current stereo
        if len(pts_2d) > 0:
            pts_3d, valid = self._triangulate_points(pts_2d, disp)
            pts_2d = pts_2d[valid]
            pts_3d = pts_3d[valid]
            track_ids = track_ids[valid]
        else:
            pts_3d = np.empty((0, 3), dtype=np.float32)

        # Detect new features if tracked count below threshold
        if len(pts_2d) < self.MIN_TRACKED_FEATURES:
            new_pts = self._detect_features_hybrid(rect_L)

            # Mevcut noktalara yakin olanlari cikar
            if len(pts_2d) > 0 and len(new_pts) > 0:
                new_pts = self._filter_near_existing(new_pts, pts_2d, min_dist=8.0)

            if len(new_pts) > 0:
                new_3d, valid = self._triangulate_points(new_pts, disp)
                new_pts = new_pts[valid]
                new_3d  = new_3d[valid]

                if len(new_pts) > 0:
                    new_ids = np.arange(
                        self._next_track_id,
                        self._next_track_id + len(new_pts),
                        dtype=np.int64
                    )
                    self._next_track_id += len(new_pts)

                    if len(pts_2d) > 0:
                        pts_2d    = np.vstack([pts_2d, new_pts.reshape(-1, 2)])
                        pts_3d    = np.vstack([pts_3d, new_3d])
                        track_ids = np.concatenate([track_ids, new_ids])
                    else:
                        pts_2d    = new_pts.reshape(-1, 2).astype(np.float32)
                        pts_3d    = new_3d
                        track_ids = new_ids

        # State'i sakla
        self._tracked_pts_2d = pts_2d if len(pts_2d) > 0 else None
        self._tracked_pts_3d = pts_3d if len(pts_3d) > 0 else None
        self._track_ids      = track_ids
        self._prev_rect_L    = rect_L

    # -- PnP odometry (SQPNP + LM Refinement) -----------------------------

    def calculate_odometry(self, pts_3d: np.ndarray, pts_2d: np.ndarray):
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
            try:
                inlier_idx = inliers.ravel()
                inlier_3d = pts_3d[inlier_idx].reshape(-1, 1, 3)
                inlier_2d = pts_2d[inlier_idx].reshape(-1, 1, 2)
                rvec, tvec = cv2.solvePnPRefineLM(
                    inlier_3d, inlier_2d,
                    self.K_rect, None, rvec, tvec
                )
            except cv2.error:
                pass
            return rvec, tvec, inliers
        return None, None, None

    # -- Kalman UPDATE step (Asama 2: 6DOF Direct Pose) --------------------

    def kalman_update_step(self, rvec: np.ndarray, tvec: np.ndarray,
                           dt: float, n_inliers: int,
                           p_prev: np.ndarray, q_prev: np.ndarray) -> str:
        """
        Asama 2: PnP rvec+tvec -> mutlak pozisyon+oryantasyon -> 6DOF ESKF update.
        v = tvec/dt KALDIRILDI. Hiz dolayli olarak P capraz korelasyonlari ile duzeltilir.

        Args:
            rvec     : PnP rotation vector (3,)
            tvec     : PnP translation vector (3,)
            dt       : frame zaman farki (s)
            n_inliers: PnP inlier sayisi
            p_prev   : IMU predict oncesi pozisyon (world frame)
            q_prev   : IMU predict oncesi quaternion (body->world)

        Returns:
            "updated"          — Kalman update basarili
            "innov_rejected"   — innovation gate reddetti
            "low_inliers"      — dusuk inlier
            "abs_thresh"       — displacement cok buyuk
            "ratio_thresh"     — oransal hiz esigi asildi
            "direction_thresh" — yon kontrolu reddetti
        """
        if n_inliers < self.MIN_INLIERS_FOR_UPDATE:
            return "low_inliers"

        # -- PnP -> Body frame -> World frame donusumu ---------------------
        # solvePnP konvansiyonu: p_cam_t1 = R_pnp @ p_obj_t0 + t_pnp
        # Kamera yer degistirmesi (t0 rectified camera frame'inde):
        #   delta_rect = -R_pnp^T @ t_pnp
        R_pnp, _ = cv2.Rodrigues(rvec.ravel())
        delta_rect = -R_pnp.T @ tvec.ravel()

        # 1) Oryantasyon: R_pnp'yi body frame'e donustur (once hesapla, lever arm icin lazim)
        #    R_delta_body = R_r2b @ R_pnp @ R_r2b^T  (SE(3) conjugation rotation part)
        R_delta_body = self._R_rect_to_body @ R_pnp @ self._R_rect_to_body.T

        # 2) Rectified camera frame -> Body (IMU) frame
        delta_body = self._R_rect_to_body @ delta_rect

        # 3) Body frame -> World frame (q_prev = body-to-world at t0)
        R_prev = quat_to_rotmat(q_prev)
        delta_world = R_prev @ delta_body

        # 4) Pseudo-global pozisyon olcumu (relative PnP birikimi)
        p_meas = p_prev.ravel() + delta_world

        # 5) Mutlak oryantasyon: R_bw(t1) = R_bw(t0) @ R_delta_body^T
        #    => q_meas = q_prev ⊗ conj(q_delta_body)
        q_delta_body = rotmat_to_quat(R_delta_body)
        q_meas = quat_mult(q_prev.ravel(), quat_conj(q_delta_body))
        q_meas = quat_normalize(q_meas)

        # -- Pre-filter: Absurt olcumleri reddet ---------------------------
        displacement = np.linalg.norm(p_meas - p_prev.ravel())
        implied_speed = displacement / max(dt, 1e-9)

        MAX_DISPLACEMENT_SPEED = 5.0
        if implied_speed > MAX_DISPLACEMENT_SPEED:
            return "abs_thresh"

        current_speed = np.linalg.norm(self.kalman.velocity)
        if current_speed > 0.1 and implied_speed > 0.1:
            speed_ratio = implied_speed / current_speed
            if speed_ratio > 4.0:
                return "ratio_thresh"

        if displacement > 0.005 and current_speed > 0.1:
            disp_dir = (p_meas - p_prev.ravel()) / displacement
            vel_dir = self.kalman.velocity / current_speed
            cos_angle = np.dot(disp_dir, vel_dir)
            if cos_angle < -0.7:
                return "direction_thresh"

        # -- 9DOF Kalman update (pozisyon + hiz + oryantasyon) --------------
        v_pseudo = delta_world / max(dt, 1e-9)
        self.kalman.update_pose(p_meas, q_meas, n_inliers, v_pseudo=v_pseudo)

        if not self.kalman.last_update_accepted:
            return "innov_rejected"
        return "updated"

    # -- Hiz surekliligi kontrolu ------------------------------------------

    def check_speed_continuity(self, speed: float, max_ratio: float = 3.0) -> bool:
        if self._prev_speed < 0.05:
            self._prev_speed = speed
            return True

        if speed > 0.05:
            ratio = speed / max(self._prev_speed, 0.01)
            if ratio > max_ratio:
                return False

        self._prev_speed = speed
        return True

    # -- Hiz erisimi (EMA filtreli) ----------------------------------------

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
