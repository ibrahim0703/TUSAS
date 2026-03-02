import cv2
import numpy as np
from config import k_left, d_left, k_right, d_right, r_matrix, t_vector, baseline

class StereoOdometryTracker:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500) # ORB yi 1 kere yaratıyoruz
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # crossCheck=True için kullanılır

        # Calculate Ideal Projection Matrices WITHOUT distorting the actual images
        # 2 kameranın aradaki farklarını yok ederek ortak eş bir kamera üstünden hesaplama ve projectionları yapmamızı sağlar.
        self.R1, self.R2, self.P1, self.P2, _ = cv2.fisheye.stereoRectify(
            k_left, d_left, k_right, d_right, (512, 512), r_matrix, t_vector, flags=cv2.CALIB_ZERO_DISPARITY
        )
        self.f_ideal = self.P1[0, 0] # The new, mathematically safe focal length

    def process_space_get_depth(self, raw_left, raw_right):
        """
        Solves the SPACE problem (Z-Depth) using Stereo Matching at time t0.
        We cannot use Optical flow LK method because it is valid for small motions but there is 11 cm distance between two cameras. 
        It causes no-matching pixel problem.
        ORB gives each pixel a "Descriptor" (ID card) to find it anywhere.
        """
        # 1. Detect and Compute Descriptors / kp -> Find corners , des -> give them ID cards
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
        # Why undistortPoints? Instead of undistorting the whole 512x512 image which causes  tearing/Mercator issues on 190-deg fisheye
        # we ONLY undistort the found coordinates.
        # This saves 99% of CPU and prevents image corruption.
        rect_l = cv2.fisheye.undistortPoints(raw_l_pts, k_left, d_left, R=self.R1, P=self.P1)
        rect_r = cv2.fisheye.undistortPoints(raw_r_pts, k_right, d_right, R=self.R2, P=self.P2)

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
                    z = (self.f_ideal * baseline) / disp
                    
                    # We store the RAW point (for temporal tracking) and its DEPTH
                    valid_raw_l.append(raw_l_pts[i])
                    valid_depths.append(z)

        # Return format: (N, 1, 2) float32 array for Lucas-Kanade compatibility
        return np.array(valid_raw_l, dtype=np.float32), valid_depths


    def track_time_get_flow(self, img_t0, img_t1, pts_t0):
        """
        Solves the TIME problem using Optical Flow.
        Why Lucas-Kanade? Because between frame t0 and t1 (50ms), movement is tiny (3-5 pixels). 
        LK is exceptionally fast and sub-pixel accurate for small motions so it is accurate from t0 to t1.
        """
        if len(pts_t0) == 0:
            return [], [], []
        
        # MOTION TRACKING: Lucas-Kanade Optical Flow
        lk_params = dict(winSize=(15, 15),        # winSize=(15, 15): Searches within a 15px radius. Perfect for high FPS.
                         maxLevel=2,              # Image ın piksellerini ne kadar küçülterek arama yapacağını söyler. piksel# = piksel# / maxLevel
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Track the points from t0 in the new image t1
        # st == 1 means the point was successfully tracked
        p1, st, err = cv2.calcOpticalFlowPyrLK(img_t0,      # ımage scene from t0 time 
                                               img_t1,      # ımage scene from t1 time 
                                               pts_t0, None,# points from t0 time
                                               **lk_params) # LK parameters dict
        valid_p0 = []
        valid_p1 = []
        movements = []
        # Tespit edilen geçerli noktaları filtreleyerek hareketi ve noktaların eski ve yeni kordinatlarını returnlüyoruz.
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
