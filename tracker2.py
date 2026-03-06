import cv2
import numpy as np
from config import K_LEFT, D_LEFT, K_RIGHT, D_RIGHT, R_MATRIX, T_VECTOR, BASELINE

class StereoOdometryTracker:
    def __init__(self):
        # 500 Nokta ve ORB Kimliği
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Sanal Projeksiyon Matrisleri
        self.R1, self.R2, self.P1, self.P2, _ = cv2.fisheye.stereoRectify(
            K_LEFT, D_LEFT, K_RIGHT, D_RIGHT, (512, 512), R_MATRIX, T_VECTOR, flags=cv2.CALIB_ZERO_DISPARITY
        )
        self.f_ideal = self.P1[0, 0]

    def process_space_get_depth(self, raw_left, raw_right):
        kp1, des1 = self.orb.detectAndCompute(raw_left, None)
        kp2, des2 = self.orb.detectAndCompute(raw_right, None)

        if des1 is None or des2 is None:
            return [], []

        matches = self.bf.match(des1, des2)
        if len(matches) == 0:
            return [], []

        raw_l_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        raw_r_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        rect_l = cv2.fisheye.undistortPoints(raw_l_pts, K_LEFT, D_LEFT, R=self.R1, P=self.P1)
        rect_r = cv2.fisheye.undistortPoints(raw_r_pts, K_RIGHT, D_RIGHT, R=self.R2, P=self.P2)

        valid_raw_l = []
        valid_3d_pts = []
        
        # [DEĞİŞİKLİK 1]: Acımasız Derinlik (Space) Filtresi
        for i in range(len(rect_l)):
            xl, yl = rect_l[i][0]
            xr, yr = rect_r[i][0]

            # Y toleransı 2.0'ye çekildi
            if abs(yl - yr) < 2.0: 
                disp = xl - xr
                # Disparity 2 pikselden büyük olmak ZORUNDA
                if disp > 2.0:
                    z = (self.f_ideal * BASELINE) / disp
                    
                    # Derinlik 0.5m ile 8.0m arasında olmak ZORUNDA
                    if 0.5 < z < 8.0:
                        X = (xl - self.P1[0, 2]) * z / self.P1[0, 0]
                        Y = (yl - self.P1[1, 2]) * z / self.P1[1, 1]
                        
                        valid_raw_l.append(raw_l_pts[i])
                        valid_3d_pts.append((X, Y, z))

        return np.array(valid_raw_l, dtype=np.float32), np.array(valid_3d_pts, dtype=np.float32)

    def track_time_get_flow(self, img_t0, img_t1, pts_t0):
        if len(pts_t0) == 0:
            return [], [], []

        lk_params = dict(winSize=(15, 15), maxLevel=2, 
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        p1, st, err = cv2.calcOpticalFlowPyrLK(img_t0, img_t1, pts_t0, None, **lk_params)

        valid_p0 = []
        valid_p1 = []
        movements = []

        for i in range(len(pts_t0)):
            if st[i] == 1:
                x0, y0 = pts_t0[i].ravel()
                x1, y1 = p1[i].ravel()

                valid_p0.append((x0, y0))
                valid_p1.append((x1, y1))

                shift = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                movements.append(shift)

        return valid_p0, valid_p1, movements

    def calculate_odometry(self, pts_3d_t0, pts_2d_t1_raw):
        # [DEĞİŞİKLİK 2]: Minimum nokta sınırı arttı (6)
        if len(pts_3d_t0) < 6:
            return None, None, None

        # [DEĞİŞİKLİK 3]: RANSAC Sıkılaştırıldı (reprojectionError=2.0)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_t0, 
            pts_2d_t1_raw, 
            K_LEFT, 
            D_LEFT,
            reprojectionError=2.0, 
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP
        )

        if success and inliers is not None:
            return rvec, tvec, inliers
        return None, None, None
