# --- ESKİSİNİ SİL, BUNU YAPIŞTIR ---
def get_roi_mask(frame):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Dikkat: Köşeli parantezlere ve array yapısına dikkat et.
    # pts shape'i (N, 1, 2) veya (N, 2) olmalı.
    pts = np.array([
        [int(w * 0.1), int(h * 0.90)],  # Sol Alt
        [int(w * 0.35), int(h * 0.45)], # Sol Üst
        [int(w * 0.65), int(h * 0.45)], # Sağ Üst
        [int(w * 0.9), int(h * 0.90)]   # Sağ Alt
    ], dtype=np.int32)

    # reshape(-1, 1, 2) OpenCV'nin en sevdiği formattır, hata vermez.
    pts = pts.reshape((-1, 1, 2))
    
    cv.fillPoly(mask, [pts], 255)
    return mask, pts
