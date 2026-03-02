import cv2
import numpy as np
from config import K_LEFT, D_LEFT, K_RIGHT, D_RIGHT, R_MATRIX, T_VECTOR, BASELINE

class StereoOdometryTracker:
    def __init__(self):
        # ======================================================================
        # SPACE (DEPTH) INITIALIZATION: ORB + BruteForce
        # Why ORB? To bridge the large 10cm gap between Left and Right cameras.
        # Optical flow fails here because the pixel disparity is too large (>50px).
        # ORB gives each pixel a "Descriptor" (ID card) to find it anywhere.
        # ======================================================================
        self.orb = cv2.ORB_create(nfeatures=500)
        # NORM_HAMMING is used for ORB binary descriptors. crossCheck=True ensures mutual matching.
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Calculate Ideal Projection Matrices WITHOUT distorting the actual images
        # Why CALIB_ZERO_DISPARITY? It aligns the optical centers horizontally.
        self.R1, self.R2, self.P1, self.P2, _ = cv2.fisheye.stereoRectify(
            K_LEFT, D_LEFT, K_RIGHT, D_RIGHT, (512, 512), R_MATRIX, T_VECTOR, flags=cv2.CALIB_ZERO_DISPARITY
        )
        self.f_ideal = self.P1[0, 0] # The new, mathematically safe focal length

    def process_space_get_depth(self, raw_left, raw_right):
        """
        Solves the SPACE problem (Z-Depth) using Stereo Matching at time t0.
        """
        # 1. Detect and Compute Descriptors (Find corners and give them ID cards)
        kp1, des1 = self.orb.detectAndCompute(raw_left, None)
        kp2, des2 = self.orb.detectAndCompute(raw_right, None)

        if des1 is None or des2 is None:
            return [], []

        # 2. Match points between Left and Right cameras using Descriptors
        matches = self.bf.match(des1, des2)
        if len(matches) == 0:
            return [], []

        # Extract raw (distorted) (x, y) coordinates of the matches
        raw_l_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        raw_r_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 3. Mathematical Undistortion
        # Why undistortPoints? Instead of undistorting the whole 512x512 image (which causes 
        # tearing/Mercator issues on 190-deg fisheye), we ONLY undistort the found coordinates.
        # This saves 99% of CPU and prevents image corruption.
        rect_l = cv2.fisheye.undistortPoints(raw_l_pts, K_LEFT, D_LEFT, R=self.R1, P=self.P1)
        rect_r = cv2.fisheye.undistortPoints(raw_r_pts, K_RIGHT, D_RIGHT, R=self.R2, P=self.P2)

        valid_raw_l = []
        valid_depths = []
        
        # 4. Epipolar Check and Triangulation
        for i in range(len(rect_l)):
            xl, yl = rect_l[i][0]
            xr, yr = rect_r[i][0]

            # Epipolar Constraint: Y coordinates must be almost identical in rectified space.
            if abs(yl - yr) < 3.0: 
                disp = xl - xr
                if disp > 0:
                    z = (self.f_ideal * BASELINE) / disp
                    
                    # We store the RAW point (for temporal tracking) and its DEPTH
                    valid_raw_l.append(raw_l_pts[i])
                    valid_depths.append(z)

        # Return format: (N, 1, 2) float32 array for Lucas-Kanade compatibility
        return np.array(valid_raw_l, dtype=np.float32), valid_depths


    def track_time_get_flow(self, img_t0, img_t1, pts_t0):
        """
        Solves the TIME problem (Motion/Odometry) using Optical Flow from t0 to t1.
        """
        if len(pts_t0) == 0:
            return [], [], []

        # ======================================================================
        # TIME (MOTION) TRACKING: Lucas-Kanade Optical Flow
        # Why Lucas-Kanade? Because between frame t0 and t1 (50ms), movement is 
        # tiny (3-5 pixels). LK is exceptionally fast and sub-pixel accurate for 
        # small motions. We don't need heavy ORB descriptors here.
        # winSize=(15, 15): Searches within a 15px radius. Perfect for high FPS.
        # ======================================================================
        lk_params = dict(winSize=(15, 15), 
                         maxLevel=2, 
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Track the points from t0 in the new image t1
        # st == 1 means the point was successfully tracked
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

                # Calculate Euclidean distance (pixel shift) between t0 and t1
                shift = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                movements.append(shift)

        return valid_p0, valid_p1, movements
