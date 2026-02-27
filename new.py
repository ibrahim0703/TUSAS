    # Döngünün içinde, çizgileri çizmeden HEMEN ÖNCE maskeyi sıfırla:
    mask = np.zeros_like(frame_color)

    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        
        # Düz çizgi yerine hareket yönünü gösteren OK çiz
        cv2.arrowedLine(mask, (c, d), (a, b), (0, 255, 0), 2, tipLength=0.3)
        frame_color = cv2.circle(frame_color, (a, b), 4, (0, 0, 255), -1)
