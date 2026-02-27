import cv2
import numpy as np
import os
import glob
import csv
from collections import deque

# --- 1. SİSTEM PARAMETRELERİ ---
f_x, f_y = 458.654, 457.296
c_x, c_y = 367.215, 248.375
baseline = 0.110 

K = np.array([[f_x,   0, c_x],
              [  0, f_y, c_y],
              [  0,   0,   1]], dtype=np.float64)
dist_coeffs = np.array([-0.2834, 0.0739, 0.00019, 0.000017])

# SGBM'YE DÖNÜŞ (Derinlik Patlamasını Önlemek İçin)
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

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- 2. MODÜLLER ---
def get_features_fast_clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    grid_x, grid_y = 4, 4
    h, w = img_clahe.shape
    cell_h, cell_w = h // grid_y, w // grid_x
    corners = []
    
    for i in range(grid_y):
        for j in range(grid_x):
            roi = img_clahe[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            kp = fast.detect(roi, None)
            if len(kp) > 0:
                kp = sorted(kp, key=lambda x: x.response, reverse=True)[:20]
                for p in kp:
                    corners.append([[np.float32(p.pt[0] + j*cell_w), np.float32(p.pt[1] + i*cell_h)]])
                    
    corners = np.array(corners)
    if len(corners) > 0:
        corners = cv2.cornerSubPix(img_clahe, corners, (5, 5), (-1, -1), subpix_criteria)
    return corners

def calculate_3d_points(p0_2d, disparity_map):
    points_3d = []
    valid_points_2d = []
    
    # Derinlik haritasının fiziksel sınırlarını al (480 Yükseklik, 752 Genişlik)
    h, w = disparity_map.shape 
    
    for pt in p0_2d:
        u, v = pt[0][0], pt[0][1] 
        
        # SINIR KONTROLÜ (BOUNDARY CHECK): 
        # Eğer piksel ekranın solundan, sağından, üstünden veya altından dışarı çıktıysa onu anında ÇÖPE AT.
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
            
        d = disparity_map[int(v), int(u)] / 16.0 
        
        # DERİNLİK FİLTRESİ
        if d > 1.0: 
            Z = (f_x * baseline) / d
            if Z > 20.0:
                continue
                
            X = ((u - c_x) * Z) / f_x
            Y = ((v - c_y) * Z) / f_y
            points_3d.append([X, Y, Z])
            valid_points_2d.append([[np.float32(u), np.float32(v)]])
            
    return np.array(points_3d, dtype=np.float32), np.array(valid_points_2d, dtype=np.float32)

# --- 3. BAŞLANGIÇ VE LOGLAMA KURULUMU ---
left_folder = 'cam0/data'  
right_folder = 'cam1/data'
left_images = sorted(glob.glob(os.path.join(left_folder, '*.png')))
right_images = sorted(glob.glob(os.path.join(right_folder, '*.png')))

# Log Dosyasını Hazırla
log_filename = "svo_velocity_log.csv"
print(f"[LOG] Veriler {log_filename} dosyasina yazdirilacak.")

with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp_ns", "Z_Velocity_m_s"]) # Başlıklar

    old_left = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
    old_right = cv2.imread(right_images[0], cv2.IMREAD_GRAYSCALE)

    old_disparity = stereo.compute(old_left, old_right)
    p0_2d_raw = get_features_fast_clahe(old_left)
    p0_3D, p0_2D = calculate_3d_points(p0_2d_raw, old_disparity)

    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    velocity_history_z = deque(maxlen=5) 
    smoothed_z_vel = 0.0

    # --- 4. ZAMAN DÖNGÜSÜ ---
    for i in range(1, len(left_images)):
        new_left = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
        new_right = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)
        
        # Dosya adından nanosaniye zaman damgasını kopar (örn: 1403636579763555584)
        timestamp = os.path.basename(left_images[i]).replace('.png', '')
        
        p1_2D, st, err = cv2.calcOpticalFlowPyrLK(old_left, new_left, p0_2D, None, **lk_params)
        
        good_new_2D = p1_2D[st == 1]
        good_old_3D = p0_3D[st.flatten() == 1]
        good_old_2D = p0_2D[st == 1] 
        
        if len(good_new_2D) > 10:
            success, R_vec, t_vec, inliers = cv2.solvePnPRansac(
                objectPoints=good_old_3D, 
                imagePoints=good_new_2D, 
                cameraMatrix=K, 
                distCoeffs=dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                raw_z_vel = -t_vec[2][0] * 20.0 
                
                if abs(raw_z_vel) < 0.05:
                    raw_z_vel = 0.0
                
                velocity_history_z.append(raw_z_vel)
                smoothed_z_vel = sum(velocity_history_z) / len(velocity_history_z)
                
                # LOG DOSYASINA YAZ
                writer.writerow([timestamp, smoothed_z_vel])
                
        # Görselleştirme
        frame_color = cv2.cvtColor(new_left, cv2.COLOR_GRAY2BGR)
        for (new, old) in zip(good_new_2D, good_old_2D):
            a, b = int(new[0]), int(new[1])
            c, d = int(old[0]), int(old[1])
            cv2.arrowedLine(frame_color, (c, d), (a, b), (0, 255, 0), 2, tipLength=0.3)
        
        text = f"Z-Velocity: {smoothed_z_vel:.2f} m/s"
        color = (0, 255, 0) if abs(smoothed_z_vel) > 0.0 else (0, 0, 255)
        cv2.putText(frame_color, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
        
        cv2.imshow('SVO: Gercekci Derinlik Filtresi & Loglama', frame_color)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
        old_left = new_left.copy()
        
        pixel_shift = np.linalg.norm(good_new_2D - good_old_2D.reshape(-1, 2), axis=1).mean()
        
        if len(good_new_2D) < 80:
            old_disparity = stereo.compute(old_left, new_right)
            p0_2d_raw = get_features_fast_clahe(old_left)
            p0_3D, p0_2D = calculate_3d_points(p0_2d_raw, old_disparity)
            
        elif pixel_shift > 3.0: 
            old_disparity = stereo.compute(old_left, new_right)
            p0_3D, p0_2D = calculate_3d_points(good_new_2D.reshape(-1, 1, 2), old_disparity)
            
        else:
            p0_2D = good_new_2D.reshape(-1, 1, 2)
            p0_3D = good_old_3D 

cv2.destroyAllWindows()
print(f"[SİSTEM] Uçuş sonlandı. Veriler {log_filename} dosyasına kaydedildi.")
