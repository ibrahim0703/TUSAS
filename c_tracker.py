import cv2
import numpy as np
from config import k_left, d_left, k_right, d_right, r_matrix, t_vector, baseline


class IMUPreintegrator:
    def __init__(self):
        self.gravity = np.array([0.0, 0.0, 9.81])
        self.reset()

    def reset(self):
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.dt_total = 0.0

    def integrate(self, gyro, accel, dt):
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

        accel_world  = self.delta_R @ accel - self.gravity
        self.delta_p += self.delta_v * dt + 0.5 * accel_world * dt**2
        self.delta_v += accel_world * dt
        self.delta_R  = self.delta_R @ dR
        self.dt_total += dt

    def get_prediction(self):
        return self.delta_R.copy(), self.delta_v.copy(), self.delta_p.copy()


class StereoOdometryTracker:
    def __init__(self):
        self.orb  = cv2.ORB_create(nfeatures=800)
        self.imu  = IMUPreintegrator()

        # stereoRectify → 7 deger: R1, R2, P1, P2, Q, roi1, roi2
        result = cv2.fisheye.stereoRectify(
                k_left, d_left, k_right, d_right,
                (512, 512), r_matrix, t_vector,
                flags=cv2.CALIB_ZERO_DISPARITY
            )
        self.R1, self.R2, self.P1, self.P2, self.Q = result[:5] 

        self.f_ideal = self.P1[0, 0]
        self.K_rect  = self.P1[:3, :3]

        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=64, blockSize=11,
            P1=8*3*11**2, P2=32*3*11**2,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self._maps_ready = False

    def _ensure_maps(self, h, w):
        if self._maps_ready:
            return
        self._ml1, self._ml2 = cv2.fisheye.initUndistortRectifyMap(
            k_left,  d_left,  self.R1, self.P1, (w, h), cv2.CV_32FC1)
        self._mr1, self._mr2 = cv2.fisheye.initUndistortRectifyMap(
            k_right, d_right, self.R2, self.P2, (w, h), cv2.CV_32FC1)
        self._maps_ready = True

    def rectify_image(self, img, side='left'):
        """Ham goruntu → rectified goruntu. side='left' veya 'right'."""
        h, w = img.shape[:2]
        self._ensure_maps(h, w)
        if side == 'left':
            return cv2.remap(img, self._ml1, self._ml2, cv2.INTER_LINEAR)
        else:
            return cv2.remap(img, self._mr1, self._mr2, cv2.INTER_LINEAR)

    def process_space_get_depth(self, raw_left, raw_right):
        """
        Donus degerleri (3 adet):
          pts_rect  : (N,1,2) float32  — rectified sol pikseller
          pts_3d    : (N,3)   float32  — 3D noktalar (metre)
          rect_l    : rectified sol goruntu (flow icin)
        """
        h, w = raw_left.shape[:2]
        self._ensure_maps(h, w)

        rect_l = cv2.remap(raw_left,  self._ml1, self._ml2, cv2.INTER_LINEAR)
        rect_r = cv2.remap(raw_right, self._mr1, self._mr2, cv2.INTER_LINEAR)

        disp = self.sgbm.compute(rect_l, rect_r).astype(np.float32) / 16.0

        kp_list, _ = self.orb.detectAndCompute(rect_l, None)
        if not kp_list:
            return np.array([]), np.array([]), rect_l

        cx, cy = self.P1[0, 2], self.P1[1, 2]
        fx, fy = self.P1[0, 0], self.P1[1, 1]

        pts_rect_list = []
        pts_3d_list   = []

        for kp in kp_list:
            xr, yr = kp.pt
            xi, yi = int(round(xr)), int(round(yr))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            d = disp[yi, xi]
            if d < 0.5:
                continue
            z = (self.f_ideal * baseline) / d
            if not (0.3 < z < 15.0):
                continue
            X = (xr - cx) * z / fx
            Y = (yr - cy) * z / fy
            pts_rect_list.append([xr, yr])
            pts_3d_list.append([X, Y, z])

        if not pts_3d_list:
            return np.array([]), np.array([]), rect_l

        pts_rect = np.array(pts_rect_list, dtype=np.float32).reshape(-1, 1, 2)
        pts_3d   = np.array(pts_3d_list,   dtype=np.float32)
        return pts_rect, pts_3d, rect_l

    def track_time_get_flow(self, rect_t0, rect_t1, pts_t0):
        if len(pts_t0) == 0:
            return [], [], []

        lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )
        p1, st, _ = cv2.calcOpticalFlowPyrLK(rect_t0, rect_t1, pts_t0, None, **lk_params)

        valid_p0, valid_p1, movements = [], [], []
        for i in range(len(pts_t0)):
            if st[i] == 1:
                x0, y0 = pts_t0[i].ravel()
                x1, y1 = p1[i].ravel()
                valid_p0.append((x0, y0))
                valid_p1.append((x1, y1))
                movements.append(np.hypot(x1 - x0, y1 - y0))
        return valid_p0, valid_p1, movements

    def calculate_odometry(self, pts_3d_t0, pts_2d_t1_rect):
        if len(pts_3d_t0) < 6:
            return None, None, None

        pts_2d = np.array(pts_2d_t1_rect, dtype=np.float32).reshape(-1, 1, 2)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_t0.reshape(-1, 1, 3),
            pts_2d,
            self.K_rect,   # rectified ideal K
            None,          # distortion YOK
            reprojectionError=2.0,
            confidence=0.99,
            iterationsCount=200,
            flags=cv2.SOLVEPNP_EPNP
        )

        if success and inliers is not None and len(inliers) >= 6:
            return rvec, tvec, inliers
        return None, None, None

    def fuse_imu_vision(self, tvec_vision, imu_delta_p, n_inliers):
        w_v   = min(0.9, 0.5 + n_inliers * 0.01)
        w_i   = 1.0 - w_v
        t_v   = tvec_vision.ravel()
        t_i   = imu_delta_p.ravel()
        fused = t_v.copy()
        fused[2] = w_v * t_v[2] + w_i * t_i[2]
        return fused.reshape(3, 1)
