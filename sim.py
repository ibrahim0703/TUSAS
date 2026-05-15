# --- KONTROLCÜLERİ YARAT ---
# Dış Döngü (Konum -> Hedef Açıyı Üretir). Çıkış limiti radyan cinsinden maks eğilme (örn: 0.5 rad)
pid_x_pos = PIDController(Kp=0.8, Ki=0.0, Kd=1.2, output_limit=0.5) 

# İç Döngü (Açı -> Motor Torkunu Üretir). Çıkış limiti Newton-metre cinsinden maks Tork
pid_pitch_att = PIDController(Kp=5.0, Ki=0.0, Kd=2.1, output_limit=2.0)

# Simülasyon Zaman Ayarları
dt_inner = 0.002 # 500 Hz (İç döngü refleks hızı)
dt_outer = 0.02  # 50 Hz (Dış döngü karar hızı)
outer_loop_counter = 0

# Hedeflerimiz
target_x = 5.0 # Metre

# Anlık Durumlar (Gerçek simülasyonda bunları fizik motorundan / sensörden okuyacaksın)
current_x = 0.0
current_pitch = 0.0
target_pitch = 0.0 

# --- ANA SİMÜLASYON DÖNGÜSÜ ---
while True:
    # 1. DIŞ DÖNGÜ (Daha yavaş çalışır. Her 10 iç döngüde 1 kez tetiklenir)
    if outer_loop_counter >= (dt_outer / dt_inner):
        # Konumdaki hataya bak, ne kadar eğilmemiz gerektiğine (Pitch) karar ver
        # Dikkat: X'te ileri gitmek için Pitch'i eksi (veya artı) yapmak gerekebilir, bu referans sistemine bağlıdır.
        target_pitch = pid_x_pos.update(target_x, current_x, dt_outer)
        outer_loop_counter = 0 # Sayacı sıfırla
        
    # 2. İÇ DÖNGÜ (Her milisaniye çalışır. Dış döngüden gelen target_pitch'i hedefler)
    # Pitch hatasını ezmek için fiziksel olarak ne kadar Tork (bükme gücü) lazım?
    torque_y = pid_pitch_att.update(target_pitch, current_pitch, dt_inner)
    
    # 3. KASLARA GÜÇ VER (Mixer ve Fizik Motoru)
    # Elde edilen torque_y, mikser matrisinden geçip motor RPM'lerine dönüşür
    # update_physics_engine(torque_y, dt_inner) ...
    
    # Zamanı ilerlet
    outer_loop_counter += 1
    # sleep(dt_inner) # Gerçek zamanlı çalışıyorsa
