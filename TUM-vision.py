import numpy as np

# ==============================================================================
# CAMERA INTRINSICS & EXTRINSICS (TUM-VI Room Dataset)
# Why separate this? Hardware changes. You shouldn't hunt for matrices inside 
# the algorithm code when you mount a different camera.
# ==============================================================================

# 1. Left Camera (cam0) Intrinsics & Distortion (Equidistant/Fisheye model)
K_LEFT = np.array([[190.978, 0.0, 254.931], 
                   [0.0, 190.973, 256.897], 
                   [0.0, 0.0, 1.0]], dtype=np.float64)
D_LEFT = np.array([0.003482, 0.000715, -0.002053, 0.0002029], dtype=np.float64)

# 2. Right Camera (cam1) Intrinsics & Distortion
K_RIGHT = np.array([[190.442, 0.0, 252.597], 
                    [0.0, 190.434, 254.917], 
                    [0.0, 0.0, 1.0]], dtype=np.float64)
D_RIGHT = np.array([0.003400, 0.001766, -0.002663, 0.0003299], dtype=np.float64)

# 3. Extrinsics: Rotation (R) and Translation (T) from Left to Right camera
R_MATRIX = np.array([[ 0.99999719,  0.00160241,  0.00174676],
                     [-0.00160269,  0.9999987 ,  0.00016067],
                     [-0.0017465 , -0.00016347,  0.99999846]], dtype=np.float64)

T_VECTOR = np.array([-0.10093155, -0.00017163, -0.00067332], dtype=np.float64)

# Stereo Baseline (Absolute distance in X-axis between cameras in meters)
BASELINE = abs(T_VECTOR[0]) # ~0.1009 meters (10 cm)
