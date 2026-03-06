import cv2
import numpy as np
import glob
import os
from tracker import StereoOdometryTracker

def main():
    print("[SYSTEM] Booting Visual Odometry Pipeline (Velocity Focus with Constraints)...")
    tracker = StereoOdometryTracker()

    # DİKKAT: Dosya yollarını kendi sistemine göre ayarla
    left_images = sorted(glob.glob('mav0/cam0/data/*.png'))
    right_images = sorted(glob.glob('mav0/cam1/data/*.png'))

    if len(left_images) == 0 or len(right_images) == 0:
        print("[FAILED] Klasörde resim bulunamadı. Dosya yollarını kontrol et.")
        return

    log_file = open("estimated_trajectory.csv", "w")
    log_file.write("timestamp,tx,ty,tz,speed_m_s\n")

    print(f"[SYSTEM] {len(left_images)} adet kare işleme alındı. Ekranda 'q' ile durdurabilirsin.")

    for i in range(len(left_images) - 1):
        t0_timestamp = int(os.path.basename(left_images[i]).split('.')[0])
        t1_timestamp = int(os.path.basename(left_images[i+1]).split('.')[0])
        dt = (t1_timestamp - t0_timestamp) * 1e-9

        img_left_t0 = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
        img_right_t0 = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)
        img_left_t1 = cv2.imread(left_images[i+1], cv2.IMREAD_GRAYSCALE)

        pts_t0, depths_3d = tracker.process_space_get_depth(img_left_t0, img_right_t0)
        
        # [BİLGİ]: En az 6 nokta geçmeli (Eskiden 10'du, şimdi RANSAC sınırına göre 6)
        if len(pts_t0) < 6:
            continue

        p0_tracked, p1_tracked, _ = tracker.track_time_get_flow(img_left_t0, img_left_t1, pts_t0)

        if len(p0_tracked) < 6:
            continue

        rvec, tvec, inliers = tracker.calculate_odometry(depths_3d, np.array(p1_tracked))

        if tvec is not None and len(inliers) > 6:
            speed = (np.linalg.norm(tvec) / dt) 

            # [DEĞİŞİKLİK 4]: Fiziksel/Kinematik Kısıtlama
            if speed > 4.0:
                 # Fizik kurallarına aykırı zıplama. Bu kareyi yoksay (loglama yapma).
                 continue 

            log_file.write(f"{t1_timestamp},{tvec[0][0]:.5f},{tvec[1][0]:.5f},{tvec[2][0]:.5f},{speed:.2f}\n")

            # GÖRSELLEŞTİRME EKRANI
            display_img = cv2.cvtColor(img_left_t1, cv2.COLOR_GRAY2BGR)
            for pt in p1_tracked:
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

            cv2.putText(display_img, f"Frame: {i} / {len(left_images)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_img, f"Speed (Est): {speed:.2f} m/s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_img, f"Inliers: {len(inliers)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("TUSAS Visual Odometry", display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    log_file.close()
    cv2.destroyAllWindows()
    print("\n[SYSTEM] Uçuş sonlandı. 'estimated_trajectory.csv' oluşturuldu.")

if __name__ == "__main__":
    main()
