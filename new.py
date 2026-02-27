
    # Step 5: Perspective-n-Point (PnP) Pose Estimation
    # Minimum 10 points are required to reliably solve the PnP equation
    if len(good_new_2D) > 10:
        # solvePnPRansac filters out optical flow outliers (bad tracking) and computes camera motion
        success, R_vec, t_vec, inliers = cv2.solvePnPRansac(
            objectPoints=good_old_3D,  # 3D metric coordinates from t=0
            imagePoints=good_new_2D,   # 2D pixel coordinates from t=1
            cameraMatrix=K,            # Intrinsic projection matrix
            distCoeffs=dist_coeffs,    # Lens distortion
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # t_vec contains the relative translation [X, Y, Z] in meters
            z_velocity = t_vec[2][0] # Forward/Backward movement
            x_velocity = t_vec[0][0] # Left/Right movement
            
            # Print raw estimated velocity (Multiplied by 20 assuming a 20Hz camera frame rate)
            print(f"Frame {i}: Z-Vel: {z_velocity*20:.2f} m/s | X-Vel: {x_velocity*20:.2f} m/s")
