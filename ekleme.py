    def update_velocity(self, measured_velocity):
        """
        Kameradan (PnP) gelen 3 Boyutlu Hız vektörü ile İvmeölçerin sapmalarını dizginler.
        """
        # İnovasyon = Kameranın ölçtüğü hız - IMU'nun tahmin ettiği hız
        innovation = measured_velocity.flatten() - self.v
        
        # H (Gözlem) Matrisi artık HIZA bakıyor (İndeks 3,4,5)
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        
        # Kalman Kazancı Hesaplama
        S = H @ self.P @ H.T + self.R_cam 
        K_gain = self.P @ H.T @ np.linalg.inv(S) 
        
        # Hataları Hesapla ve Enjekte Et
        error_state = K_gain @ innovation
        
        self.p += error_state[0:3]
        self.v += error_state[3:6]
        
        # Belirsizliği Güncelle
        self.P = (np.eye(15) - K_gain @ H) @ self.P


            if success:
                # 1. PnP'den gelen yer değiştirmeyi (tvec), geçen süreye bölerek 3 Boyutlu HIZ Vektörünü bul
                if kf_time_elapsed > 0:
                    v_cam_measured = tvec.flatten() / kf_time_elapsed
                    
                    # 2. Kamera eksenindeki bu hızı, IMU eksenine çevir (Elma ile Elmayı kıyaslamak için)
                    v_imu_measured = R_CB.T @ v_cam_measured
                    
                    # 3. ESKF'yi HIZ ile güncelle (O sonsuza giden ivmeyi durdur!)
                    filter_eskf.update_velocity(v_imu_measured)
                    
                kalman_speed = filter_eskf.get_speed()
                
                cv.putText(vis_frame, f"ESKF Hiz: {kalman_speed:.2f} m/s", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(vis_frame, f"GT Hiz  : {true_speed:.2f} m/s", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.putText(vis_frame, f"HATA    : {abs(kalman_speed - true_speed):.2f} m/s", (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
