    # ... (Video açma kısmı) ...
    cap = cv.VideoCapture(CONFIG['VIDEO_SOURCE'])
    
    if not cap.isOpened():
        print("!!! KRİTİK HATA !!! Video açılamadı.")
        return

    # --- YENİ EKLENECEK KISIM (OTOMATİK FPS) ---
    real_fps = cap.get(cv.CAP_PROP_FPS)
    
    # Bazen fps 0 veya nan dönebilir (güvenlik önlemi)
    if real_fps > 0 and not np.isnan(real_fps):
        CONFIG['DT'] = 1.0 / real_fps
        print(f"[BILGI] Video FPS tespit edildi: {real_fps:.2f}")
        print(f"[BILGI] Yeni DT (Delta Time): {CONFIG['DT']:.4f} saniye")
    else:
        print("[UYARI] FPS okunamadı! Varsayılan 30 FPS kullanılıyor.")
        CONFIG['DT'] = 1.0 / 30.0
    # -------------------------------------------
def main():
    # 1. Sistem Başlatma
    print(f"[BASLIYOR] Video Yolu: {CONFIG['VIDEO_SOURCE']}")
    cap = cv.VideoCapture(CONFIG['VIDEO_SOURCE'])
    
    if not cap.isOpened():
        print("!!! KRİTİK HATA !!! Video açılamadı.")
        return

    # --- OTOMATİK FPS ALGILAMA ---
    real_fps = cap.get(cv.CAP_PROP_FPS)
    if real_fps > 0 and not np.isnan(real_fps):
        CONFIG['DT'] = 1.0 / real_fps
        print(f"[SİSTEM] Video FPS: {real_fps:.2f} | Döngü Süresi (DT): {CONFIG['DT']:.4f}s")
    else:
        print("[UYARI] FPS algılanamadı, varsayılan 30 FPS kullanılıyor.")
        CONFIG['DT'] = 1.0 / 30.0

    sensors = DroneSensorInterface()

    ret, old_frame = cap.read()
    if not ret: 
        print("!!! HATA !!! İlk kare okunamadı.")
        return

    # Ön İşleme
    old_frame = cv.resize(old_frame, None, fx=CONFIG['SCALE_FACTOR'], fy=CONFIG['SCALE_FACTOR'])
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = generate_grid_features(old_gray, step=40)

    print("[INFO] Analiz Başlıyor...")

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Video bitti. Başa sarılıyor...")
            cap.set(cv.CAP_PROP_POS_FRAMES, 0) # Videoyu başa sarıp sonsuz döngü yapalım
            p0 = None # Noktaları sıfırla
            continue

        telemetry = sensors.get_telemetry_packet()
        current_agl = telemetry['lidar_agl']
        current_gyro = telemetry['imu_gyro']

        frame = cv.resize(frame, None, fx=CONFIG['SCALE_FACTOR'], fy=CONFIG['SCALE_FACTOR'])
        vis_frame = frame.copy()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if p0 is None or len(p0) < 50:
            p0 = generate_grid_features(frame_gray, step=40)
            old_gray = frame_gray.copy()
        else:
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **CONFIG['LK_PARAMS'])

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # RANSAC
                M, mask = cv.findHomography(good_old, good_new, cv.RANSAC, CONFIG['RANSAC_THRESHOLD'])

                if M is not None:
                    matchesMask = mask.ravel().tolist()
                    inlier_vectors_x = []
                    inlier_vectors_y = []

                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        if matchesMask[i]:
                            dx = new[0] - old[0]
                            dy = new[1] - old[1]
                            inlier_vectors_x.append(dx)
                            inlier_vectors_y.append(dy)
                            cv.line(vis_frame, (int(new[0]), int(new[1])), (int(old[0]), int(old[1])), (0, 255, 0), 1)
                        else:
                            cv.circle(vis_frame, (int(new[0]), int(new[1])), 2, (0, 0, 255), -1)

                    if len(inlier_vectors_x) > 0:
                        mean_flow_x = np.mean(inlier_vectors_x)
                        mean_flow_y = np.mean(inlier_vectors_y)

                        # BURADA ARTIK OTOMATİK HESAPLANAN CONFIG['DT'] KULLANILIYOR
                        vx_metric = compensated_flow_to_metric_velocity(mean_flow_x, current_agl, CONFIG['FOCAL_LENGTH_PX'], CONFIG['DT'], current_gyro)
                        vy_metric = compensated_flow_to_metric_velocity(mean_flow_y, current_agl, CONFIG['FOCAL_LENGTH_PX'], CONFIG['DT'], current_gyro)

                        v_forward = vy_metric
                        v_right = vx_metric

                        color_speed = (0, 255, 0) if v_forward > 0 else (0, 0, 255)
                        cv.putText(vis_frame, f"HIZ (Ileri): {v_forward:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, color_speed, 2)
                        
                        h, w = frame.shape[:2]
                        center = (w // 2, h // 2)
                        end_pt = (int(center[0] - mean_flow_x * 20), int(center[1] - mean_flow_y * 20))
                        cv.arrowedLine(vis_frame, center, end_pt, (0, 255, 255), 4, tipLength=0.3)

                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                     old_gray = frame_gray.copy()
                     p0 = good_new.reshape(-1, 1, 2)
            else:
                 p0 = generate_grid_features(frame_gray, step=40)
                 old_gray = frame_gray.copy()

        cv.putText(vis_frame, f"AGL: {current_agl:.1f} m | FPS: {real_fps:.0f}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv.imshow('TUSAS Optical Flow Module (Auto FPS)', vis_frame)
        
        # Bekleme süresini de FPS'e göre dinamik yapıyoruz
        delay = int(1000 / real_fps) if real_fps > 0 else 30
        if cv.waitKey(delay) & 0xff == 27: 
            break

    cap.release()
    cv.destroyAllWindows()
