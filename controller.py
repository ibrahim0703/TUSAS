class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limit):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        # Fiziksel sınır (Motorların satüre olmasını engellemek için)
        self.output_limit = output_limit 
        
        # Hafıza (Geçmişi hatırlaması lazım)
        self.integral_error = 0.0
        self.prev_error = 0.0

    def update(self, setpoint, measured_value, dt):
        # 1. Hatayı Bul (Error Detection Noktası)
        error = setpoint - measured_value
        
        # 2. Oransal (P) - Sanal Yay
        P = self.Kp * error
        
        # 3. İntegral (I) - Hata Biriktirici (dt ile çarpılır!)
        self.integral_error += error * dt
        I = self.Ki * self.integral_error
        
        # 4. Türev (D) - Sanal Fren (dt'ye bölünür!)
        # Sıfıra bölme hatasından kaçınmak için ufak bir kontrol
        if dt > 0.0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
            
        D = self.Kd * derivative
        
        # 5. Toplam Komut (U)
        output = P + I + D
        
        # --- ÖLÜMCÜL KORUMA: ANTI-WINDUP & SATURATION ---
        if output > self.output_limit:
            output = self.output_limit
            # İntegral şişmesini durdur (Clamping)
            self.integral_error -= error * dt 
        elif output < -self.output_limit:
            output = -self.output_limit
            self.integral_error -= error * dt
            
        # Hafızayı bir sonraki döngü için güncelle
        self.prev_error = error
        
        return output
