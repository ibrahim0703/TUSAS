import cv2
import numpy as np
from tracker import StereoOdometryTracker

def main():
    print("[SYSTEM] Booting Visual Odometry Pipeline...")
    tracker = StereoOdometryTracker()
    
    # DATA LOADING
    path_left_t0 = 'cam0/data/1520530308199447626.png'
    path_right_t0 = 'cam1/data/1520530308199447626.png'
    path_left_t1 = 'cam0/data/1520530308249448626.png' 

    img_left_t0 = cv2.imread(path_left_t0, cv2.IMREAD_GRAYSCALE)
    img_right_t0 = cv2.imread(path_right_t0, cv2.IMREAD_GRAYSCALE)
    img_left_t1 = cv2.imread(path_left_t1, cv2.IMREAD_GRAYSCALE)

    # STAGE 1: SOLVE SPACE (Get 3D Depth using ORB)
    # ======================================================================
    print("\n[STAGE 1] Executing Stereo Matching (ORB)...")
    pts_t0, depths = tracker.process_space_get_depth(img_left_t0, img_right_t0)
    
    if len(pts_t0) == 0:
        print("[FAILED] No valid stereo matches found.")
        return
        
    print(f" -> Found {len(depths)} valid points with verified Z-Depth.")
    print(f" -> Average Distance to environment: {np.mean(depths):.2f} meters.")

    # STAGE 2: SOLVE TIME (Get Temporal Motion using optical flow LK method) 
    # ======================================================================
    print("\n[STAGE 2] Executing Temporal Tracking (Lucas-Kanade)...")
    p0_tracked, p1_tracked, pixel_shifts = tracker.track_time_get_flow(img_left_t0, img_left_t1, pts_t0)
    
    if len(p0_tracked) > 0:
        print(f" -> Successfully tracked {len(p0_tracked)} points into the next frame.")
        print(f" -> Average pixel movement (t0 -> t1): {np.mean(pixel_shifts):.2f} pixels.")
    else:
        print("[FAILED] Lost all points during temporal tracking.")

if __name__ == "__main__":
    main()
