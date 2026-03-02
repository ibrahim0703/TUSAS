import cv2
import numpy as np
from tracker import StereoOdometryTracker

def main():
    print("[SYSTEM] Booting Visual Odometry Pipeline...")
    tracker = StereoOdometryTracker()
    
    # ======================================================================
    # DATA LOADING
    # You MUST update these paths to match your local dataset structure!
    # t0: Current time step
    # t1: Next time step (e.g., 50 milliseconds later)
    # ======================================================================
    path_left_t0 = 'SVO_Veri/cam0/data/1520530308199447626.png'
    path_right_t0 = 'SVO_Veri/cam1/data/1520530308199447626.png'
    
    # Replace this with the VERY NEXT image in your cam0 folder
    path_left_t1 = 'SVO_Veri/cam0/data/NEXT_IMAGE_HERE.png' 

    img_left_t0 = cv2.imread(path_left_t0, cv2.IMREAD_GRAYSCALE)
    img_right_t0 = cv2.imread(path_right_t0, cv2.IMREAD_GRAYSCALE)
    img_left_t1 = cv2.imread(path_left_t1, cv2.IMREAD_GRAYSCALE)

    if img_left_t0 is None or img_left_t1 is None:
        print("[ERROR] Could not load images. Check your file paths in main.py.")
        return

    # ======================================================================
    # STAGE 1: SOLVE SPACE (Get 3D Depth)
    # ======================================================================
    print("\n[STAGE 1] Executing Stereo Matching (ORB)...")
    pts_t0, depths = tracker.process_space_get_depth(img_left_t0, img_right_t0)
    
    if len(pts_t0) == 0:
        print("[FAILED] No valid stereo matches found.")
        return
        
    print(f" -> Found {len(depths)} valid points with verified Z-Depth.")
    print(f" -> Average Distance to environment: {np.mean(depths):.2f} meters.")

    # ======================================================================
    # STAGE 2: SOLVE TIME (Get Temporal Motion)
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
