import cv2
import numpy as np
import os
import glob
import csv
from collections import deque

# ====================================================================
# 1. SYSTEM PARAMETERS & CALIBRATION (KITTI Dataset)
# ====================================================================
# Extracted from calib_cam_to_cam.txt (P_rect_00 and P_rect_01)
f_x, f_y = 721.5377, 721.5377
c_x, c_y = 609.5593, 172.8540

# Baseline = Tx / f_x = 387.5744 / 721.5377 = 0.53715 meters
baseline = 0.53715
fps = 10.0  # KITTI operates at 10 Hz (10 frames per second)

# Camera Intrinsic Matrix
K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]], dtype=np.float64)

# KITTI images are rectified; lens distortion is zero.
dist_coeffs = np.zeros(4)

# SGBM (Stereo Semi-Global Block Matching) for robust depth map generation
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=11,
    P1=8 * 1 * 11 ** 2,
    P2=32 * 1 * 11 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Sub-pixel refinement criteria
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ====================================================================
# 2. ALGORITHMIC MODULES
# ====================================================================
def get_features_gftt(img_gray):
    """
    Ana otonomi boru hattı için optimize edilmiş Shi-Tomasi (GFTT) çıkarıcı.
    Mikro-ROI maskeleme ile anlık dalgalanmaları (jitter) önlerken,
    PnP'nin geometrik yayılımını (spread) korur.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # ==========================================================
    # MİKRO ROI MASKELEME (CERRAHİ KESİM)
    # ==========================================================
    h, w = img_clahe.shape
    # Üstten sadece 40 pikseli kes (Sonsuz gökyüzünü at, binaları bırak)
    img_clahe[0:40, :] = 0
    # Alttan sadece 50 pikseli kes (Sadece kaput ve en ağır motion blur)
    img_clahe[h-50:h, :] = 0

    # Shi-Tomasi Parametreleri
    gftt_params = dict(maxCorners=20,
                       qualityLevel=0.015,
                       minDistance=7,
                       blockSize=7)

    grid_x, grid_y = 4, 4
    cell_h, cell_w = h // grid_y, w // grid_x
    corners = []

    for i in range(grid_y):
        for j in range(grid_x):
            # Maskelenmiş (siyaha boyanmış) alanlarda Shi-Tomasi hiçbir şey bulamayacaktır
            roi = img_clahe[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]

            kp = cv2.goodFeaturesToTrack(roi, **gftt_params)

            if kp is not None:
                for p in kp:
                    x, y = p.ravel()
                    global_x = np.float32(x + j * cell_w)
                    global_y = np.float32(y + i * cell_h)
                    corners.append([[global_x, global_y]])

    corners_array = np.array(corners, dtype=np.float32)

    # Alt-Piksel İyileştirmesi (Jitter'ı engellemek için zorunlu)
    if len(corners_array) > 0:
        cv2.cornerSubPix(img_clahe, corners_array, (5, 5), (-1, -1), subpix_criteria)

    return corners_array


def calculate_3d_points(p0_2d, disparity_map):
    """
    Triangulates 3D metrics from 2D pixels. Includes rigorous Boundary and Depth checks.
    """
    points_3d = []
    valid_points_2d = []
    h, w = disparity_map.shape

    for pt in p0_2d:
        u_float, v_float = pt[0][0], pt[0][1]
        u, v = int(u_float), int(v_float)

        # CRITICAL FIX: Boundary Check (Prevents 'IndexError: out of bounds')
        # If the optical flow tracked a point outside the image frame, discard it immediately.
        if u < 0 or u >= w or v < 0 or v >= h:
            continue

        d = disparity_map[v, u] / 16.0

        # DEPTH FILTER: Discard invalid disparities and points that are too far away.
        # In autonomous driving (KITTI), points beyond 25 meters introduce massive scale drift.
        if d > 1.0:
            Z = (f_x * baseline) / d
            if Z > 45.0:  # Max depth horizon: 25 meters
                continue

            X = ((u_float - c_x) * Z) / f_x
            Y = ((v_float - c_y) * Z) / f_y

            points_3d.append([X, Y, Z])
            valid_points_2d.append([[np.float32(u_float), np.float32(v_float)]])

    return np.array(points_3d, dtype=np.float32), np.array(valid_points_2d, dtype=np.float32)


# ====================================================================
# 3. INITIALIZATION & LOGGING
# ====================================================================
# UPDATE THESE PATHS TO YOUR KITTI DIRECTORY
left_folder = 'drive_data/image_00/data'
right_folder = 'drive_data/image_01/data'
left_images = sorted(glob.glob(os.path.join(left_folder, '*.png')))
right_images = sorted(glob.glob(os.path.join(right_folder, '*.png')))

if not left_images or not right_images:
    print("[ERROR] KITTI images not found. Check your folder paths!")
    exit()

log_filename = "svo_velocity_log.csv"
print(f"[SYSTEM] Pipeline started. Logging velocities to {log_filename}")

with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame_ID", "Z_Velocity_m_s"])

    old_left = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
    old_right = cv2.imread(right_images[0], cv2.IMREAD_GRAYSCALE)

    old_disparity = stereo.compute(old_left, old_right)
    p0_2d_raw = get_features_gftt(old_left)
    p0_3D, p0_2D = calculate_3d_points(p0_2d_raw, old_disparity)

    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # ADVANCED BLOCK 2: Sliding Window for Velocity Smoothing (LBA Simulation)
    velocity_history_z = deque(maxlen=5)
    smoothed_z_vel = 0.0

    # ====================================================================
    # 4. TEMPORAL LOOP (Optical Flow & PnP)
    # ====================================================================
    for i in range(1, len(left_images)):
        new_left = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
        new_right = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)

        frame_id = os.path.basename(left_images[i]).replace('.png', '')

        p1_2D, st, err = cv2.calcOpticalFlowPyrLK(old_left, new_left, p0_2D, None, **lk_params)

        good_new_2D = p1_2D[st == 1]
        good_old_3D = p0_3D[st.flatten() == 1]
        good_old_2D = p0_2D[st == 1]

        # RANSAC'I SIKILAŞTIRMA (GEOMETRİK FİLTRELEME)
        # Minimum 10 nokta kuralı hala geçerli
        if len(good_new_2D) > 10:
            success, R_vec, t_vec, inliers = cv2.solvePnPRansac(
                objectPoints=good_old_3D,
                imagePoints=good_new_2D,
                cameraMatrix=K,
                distCoeffs=dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                iterationsCount=200,  # Varsayılan 100'dür. 200 yaparak denklemi daha çok test etmesini sağlıyoruz.
                reprojectionError=2.0,
                # KRİTİK HAMLE: 8.0 olan toleransı 2.0 piksele çektik. Hareketli araba affedilmez.
                confidence=0.99  # %99 emin olmadan modeli kabul etme.
            )


            if success:
                # Multiply by FPS (10 Hz for KITTI) to convert per-frame translation to meters/second
                # -t_vec because PnP returns world motion relative to camera. We want camera motion.
                raw_z_vel = -t_vec[2][0] * fps

                # DEADZONE FILTER: Suppress micro-vibrations when stationary
                if abs(raw_z_vel) < 0.1:
                    raw_z_vel = 0.0

                velocity_history_z.append(raw_z_vel)
                smoothed_z_vel = sum(velocity_history_z) / len(velocity_history_z)

                writer.writerow([frame_id, smoothed_z_vel])

        # VISUALIZATION (HUD)
        frame_color = cv2.cvtColor(new_left, cv2.COLOR_GRAY2BGR)
        for (new, old) in zip(good_new_2D, good_old_2D):
            a, b = int(new[0]), int(new[1])
            c, d = int(old[0]), int(old[1])
            cv2.arrowedLine(frame_color, (c, d), (a, b), (0, 255, 0), 2, tipLength=0.3)

        text = f"Velocity: {smoothed_z_vel:.2f} m/s"
        color = (0, 255, 0) if abs(smoothed_z_vel) > 0.0 else (0, 0, 255)
        cv2.putText(frame_color, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
        cv2.imshow('KITTI SVO: Autonomous Vehicle Pipeline', frame_color)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        old_left = new_left.copy()

        # ADVANCED BLOCK 3: Keyframe Logic
        # Calculate average pixel movement. If the car didn't move much, don't waste CPU on new depth mapping.
        pixel_shift = np.linalg.norm(good_new_2D - good_old_2D.reshape(-1, 2), axis=1).mean()

        if len(good_new_2D) < 100:
            # Tracking lost -> Hard reset
            old_disparity = stereo.compute(old_left, new_right)
            p0_2d_raw = get_features_gftt(old_left)
            p0_3D, p0_2D = calculate_3d_points(p0_2d_raw, old_disparity)

        elif pixel_shift > 3.0:
            # Substantial movement -> Update Keyframe (Triangulate fresh 3D points)
            old_disparity = stereo.compute(old_left, new_right)
            p0_3D, p0_2D = calculate_3d_points(good_new_2D.reshape(-1, 1, 2), old_disparity)

        else:
            # Stopped or slow -> Keep old 3D points, just update 2D tracking
            p0_2D = good_new_2D.reshape(-1, 1, 2)
            p0_3D = good_old_3D

cv2.destroyAllWindows()
print(f"[SYSTEM] Run completed. Analyze the ground truth against {log_filename}")
