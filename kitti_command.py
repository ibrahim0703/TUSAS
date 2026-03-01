import cv2
import numpy as np
import os
import glob

# 1. SYSTEM PARAMETERS & CALIBRATION (EuRoC Dataset)
# ====================================================================
# Intrinsic camera parameters: Focal lengths (f_x, f_y) and Principal Points (c_x, c_y)
# These convert 2D pixel coordinates into 3D metric rays.
f_x, f_y = 458.654, 457.296
c_x, c_y = 367.215, 248.375
baseline = 0.113 # Physical distance between left and right cameras in meters (11.3 cm)

# Camera intrinsic matrix (K) required by PnP solver
K = np.array([[f_x,   0, c_x],
              [  0, f_y, c_y],
              [  0,   0,   1]], dtype=np.float64)

# Distortion coefficients (Radial and Tangential) to un-bend the lens effect
dist_coeffs = np.array([-0.2834, 0.0739, 0.00019, 0.000017])

# 2. ALGORITHMIC MODULES
# ====================================================================
# Stereo Block Matching (BM) is used instead of SGBM for real-time speed. SGBM can be used if do u want.
stereo = cv2.StereoBM_create(numDisparities=64, # numDisparities: Max pixel shift expected.
                             blockSize=15) # blockSize: Window size for matching.

def get_features_fast_clahe(img_gray):
    """
    Extracts robust 2D features using CLAHE for contrast enhancement and 
    Grid-based (Bucketing) FAST algorithm to ensure homogeneous feature distribution.
    """
    # Contrast Limited Adaptive Histogram Equalization to illuminate dark regions
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    # FAST detector is extremely fast, ideal for high-FPS robotic vision
    fast = cv2.FastFeatureDetector_create(threshold=15, 
                                          nonmaxSuppression=True)
    
    grid_x, grid_y = 4, 4 # Divide image into 4x4 grid
    h, w = img_clahe.shape
    cell_h, cell_w = h // grid_y, w // grid_x
    corners = []
    
    for i in range(grid_y):
        for j in range(grid_x):
            # Region of Interest (ROI) for the current grid cell
            roi = img_clahe[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            kp = fast.detect(roi, None)
            
            if len(kp) > 0:
                # Sort features by response (strength) and keep only the top 20 per cell
                kp = sorted(kp, key=lambda x: x.response, reverse=True)[:20]
                for p in kp:
                    # Apply offset to map local ROI coordinates back to the global image
                    corners.append([[np.float32(p.pt[0] + j*cell_w), np.float32(p.pt[1] + i*cell_h)]])
                    
    return np.array(corners)

def calculate_3d_points(p0_2d, disparity_map):
    """
    Triangulates 3D spatial coordinates (X, Y, Z) in meters ONLY for the 
    extracted 2D feature points, saving massive CPU cycles compared to dense depth maps.
    """
    points_3d = []
    valid_points_2d = []
    
    for pt in p0_2d:
        u, v = int(pt[0][0]), int(pt[0][1])
        
        # StereoBM returns disparity multiplied by 16. Divide by 16 to get true sub-pixel disparity.
        d = disparity_map[v, u] / 16.0 
        
        # Filter out invalid disparities (d <= 1.0 means infinite distance or tracking failure)
        if d > 1.0:
            # Epipolar geometry triangulation
            Z = (f_x * baseline) / d          # Depth in meters
            X = ((u - c_x) * Z) / f_x         # Lateral position in meters
            Y = ((v - c_y) * Z) / f_y         # Vertical position in meters
            
            points_3d.append([X, Y, Z])
            valid_points_2d.append([[np.float32(u), np.float32(v)]])
            
    return np.array(points_3d, dtype=np.float32), np.array(valid_points_2d, dtype=np.float32)

# 3. INITIALIZATION (t=0)
# ====================================================================
# UPDATE DIRECTORY PATHS TO MATCH YOUR LOCAL FOLDERS
left_folder = 'cam0/data'  
right_folder = 'cam1/data'

left_images = sorted(glob.glob(os.path.join(left_folder, '*.png')))
right_images = sorted(glob.glob(os.path.join(right_folder, '*.png')))

print("[SYSTEM] Initializing SVO Pipeline. Fusing Past and Future...")

# Load the very first stereo pair (t=0)
old_left = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
old_right = cv2.imread(right_images[0], cv2.IMREAD_GRAYSCALE)

# Step 1: Compute initial depth map with SM algorithm (disparity)
old_disparity = stereo.compute(old_left, old_right)

# Step 2: Extract 2D features (corners) from the left camera
p0_2d_raw = get_features_fast_clahe(old_left)

# Step 3: Triangulate initial features into 3D metric space
p0_3D, p0_2D = calculate_3d_points(p0_2d_raw, # 2d coordinates of corner > it is obtained from fast + clahe method
                                   old_disparity) # Depth values of valid corner's locations

# Lucas-Kanade Optical Flow parameters
lk_params = dict(winSize=(21, 21),  # winSize: Search area.
                 maxLevel=3, # maxLevel: Image pyramid depth for large motions. It divide image > // pixel_number = pixel_number / maxLevel
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# 4. TEMPORAL LOOP (Real-Time Video Stream & Pose Estimation)
# ====================================================================
for i in range(1, len(left_images)):
    # Read the current stereo pair (t=1)
    new_left = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
    new_right = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)
    
    # Step 4: Optical Flow. Where did the t=0 pixels move to in t=1? It returns P1 which is new location of corners (2D)
    p1_2D, st, err = cv2.calcOpticalFlowPyrLK(old_left, # left cam old ımage scene
                                              new_left, # left cam new ımage scene
                                              p0_2D, # old location of corners (2D)
                                              None, 
                                              **lk_params) # lucas kanade params
    
    # Filter only successfully tracked features (Status == 1) 
    good_new_2D = p1_2D[st == 1]
    good_old_3D = p0_3D[st.flatten() == 1]
    
    # Step 5: Perspective-n-Point (PnP) Pose Estimation
    # Minimum 10 points are required to reliably solve the PnP equation
    if len(good_new_2D) > 10:
        # solvePnPRansac filters out optical flow outliers (bad tracking) and computes camera motion
        success, R_vec, t_vec, inliers = cv2.solvePnPRansac(
            objectPoints=good_old_3D,  # 3D metric coordinates from old ımage scene
            imagePoints=good_new_2D,   # 2D pixel coordinates from new ımage scene
            cameraMatrix=K,            # Intrinsic projection matrix 3d den 2d ye nasıl geçiş için matrix
            distCoeffs=dist_coeffs,    # Lens distortion
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # t_vec contains the relative translation [X, Y, Z] in meters
            z_velocity = t_vec[2][0] # Forward/Backward movement
            x_velocity = t_vec[0][0] # Left/Right movement
            
            # Print raw estimated velocity (Multiplied by 20 assuming a 20Hz camera frame rate)
            print(f"Frame {i}: Z-Vel: {z_velocity*20:.2f} m/s | X-Vel: {x_velocity*20:.2f} m/s")
            
    # Visualization: Draw Optical Flow vectors
    frame_color = cv2.cvtColor(new_left, cv2.COLOR_GRAY2BGR)
    for (new, old) in zip(good_new_2D, p0_2D[st == 1]):
        a, b = int(new[0]), int(new[1])
        c, d = int(old[0]), int(old[1])
        cv2.arrowedLine(frame_color, (c, d), (a, b), (0, 255, 0), 2, tipLength=0.3)
    
    cv2.imshow('SVO: Optical Flow & PnP', frame_color)
    if cv2.waitKey(1) & 0xFF == 27: # Press ESC to exit
        break
        
    # --- TEMPORAL SHIFT: Future becomes the Past ---
    old_left = new_left.copy()
    
    # Resilience Protocol: Re-initialize if the number of tracked features drops below threshold
    if len(good_new_2D) < 80:
        print("[WARNING] Tracking lost or feature depletion. Re-initializing...")
        old_disparity = stereo.compute(old_left, new_right)
        p0_2d_raw = get_features_fast_clahe(old_left)
        p0_3D, p0_2D = calculate_3d_points(p0_2d_raw, old_disparity)
    else:
        # Calculate fresh 3D coordinates for the currently tracked 2D points to be used in the next loop
        old_disparity = stereo.compute(old_left, new_right)
        p0_3D, p0_2D = calculate_3d_points(good_new_2D.reshape(-1, 1, 2), old_disparity)

cv2.destroyAllWindows()
