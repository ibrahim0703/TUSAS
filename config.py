import numpy as np
import cv2 as cv

CONFIG = {
    'CAM0_CSV': 'cam0/data.csv',
    'CAM0_DIR': 'cam0/data',
    'CAM1_CSV': 'cam1/data.csv',
    'CAM1_DIR': 'cam1/data',
    'IMU_CSV_PATH': 'imu0/data.csv',
    'GT_CSV_PATH': 'state_groundtruth_estimate0/data.csv',
    'FOCAL_LENGTH_PX': 458.654,
    'CX': 376.0, 
    'CY': 240.0, 
    'BASELINE_M': 0.11,
    'LK_PARAMS': dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
}

K = np.array([[CONFIG['FOCAL_LENGTH_PX'], 0, CONFIG['CX']],
              [0, CONFIG['FOCAL_LENGTH_PX'], CONFIG['CY']],
              [0, 0, 1]], dtype=np.float64)

# EuRoC cam0 -> IMU Extrinsic Matrisi (T_BS)
T_BS = np.array([
    [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
    [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
    [-0.0257744366974, 0.00375618835797, 0.999660727108, 0.00981073058949],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float64)

R_CB = T_BS[:3, :3].T # Kamera Eksenine Geçiş

