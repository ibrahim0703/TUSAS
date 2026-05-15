import time

# ... (PID Sınıfı ve değişken tanımlamaları aynı kalacak) ...

# Simülasyon ömrü: Sadece 3 saniye çalışsın
sim_time = 0.0
max_sim_time = 3.0 

print("Simülasyon Başlıyor...\n")
print(f"{'ZAMAN (s)':<10} | {'HEDEF X':<10} | {'GÜNCEL X':<10} | {'HEDEF PITCH':<15} | {'ÜRETİLEN TORK':<15}")
print("-" * 75)

while sim_time < max_sim_time:
    # 1. DIŞ DÖNGÜ (50 Hz)
    if outer_loop_counter >= (dt_outer / dt_inner):
        target_pitch = pid_x_pos.update(target_x, current_x, dt_outer)
        outer_loop_counter = 0
        
    # 2. İÇ DÖNGÜ (500 Hz)
    torque_y = pid_pitch_att.update(target_pitch, current_pitch, dt_inner)
    
    # 3. FİZİKSEL GERÇEKLİK SİMÜLASYONU (Basit Kütle Modeli)
    # Tork -> Pitch açısını değiştirir (Basit ivmelenme)
    # Pitch -> X konumunu değiştirir (Basit kinematik)
    # Not: Gerçekte buraya fizik motoru formülleri gelir. Şimdilik torkun x'i değiştirdiğini farz edelim:
    current_pitch += torque_y * 0.01 
    current_x += current_pitch * 0.05 
    
    # Her 50 iç döngüde bir (0.1 saniyede bir) ekrana yazdır ki terminal kilitlenmesin
    if int(sim_time * 1000) % 100 == 0:
        print(f"{sim_time:<10.3f} | {target_x:<10.1f} | {current_x:<10.3f} | {target_pitch:<15.3f} | {torque_y:<15.3f}")
    
    # Zamanı ilerlet
    outer_loop_counter += 1
    sim_time += dt_inner
    
    # Gerçek zamanlı akmasını istiyorsan aşağıdaki sleep'i aç. 
    # Hızlıca bitip sonucu görmek istiyorsan kapalı kalsın.
    # time.sleep(dt_inner) 

print("\nSimülasyon Bitti.")
