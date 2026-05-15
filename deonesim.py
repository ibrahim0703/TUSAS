import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import control as ct

# --- DRONE FİZİKSEL PARAMETRELERİ (Basitleştirilmiş Pitch Ekseni) ---
Iyy = 0.01  # Eylemsizlik Momenti (Hantallık)
b = 0.005   # Hava Sürtünmesi (Neredeyse sıfır)

# Laplace Değişkeni
s = ct.tf('s')

# Drone Bedeninin Transfer Fonksiyonu G(s) = 1 / (Iyy*s^2 + b*s)
G = 1 / (Iyy * s**2 + b * s)

# Arayüz Ayarları
fig, (ax_splane, ax_time) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.35, wspace=0.3)

# Başlangıç Kazançları
initial_Kp = 0.5
initial_Kd = 0.05

def calc_system(Kp, Kd):
    """PID (PD) ile sistemi kapatıp kökleri ve zaman yanıtını hesaplar"""
    # Kontrolcü C(s) = Kp + Kd*s
    C = Kp + Kd * s
    # Kapalı Çevrim T(s) = (C*G) / (1 + C*G)
    T = ct.feedback(C * G)
    
    poles = ct.pole(T)
    time, response = ct.step_response(T, T=np.linspace(0, 3, 500))
    
    # Zeta ve Wn hesaplama
    wn = np.sqrt(Kp / Iyy)
    zeta = (b + Kd) / (2 * Iyy * wn)
    
    return poles, time, response, zeta, wn

poles, t, y, zeta, wn = calc_system(initial_Kp, initial_Kd)

# --- SOL GRAFİK: S-DÜZLEMİ (KÖKLER) ---
p_plot, = ax_splane.plot(np.real(poles), np.imag(poles), 'rx', markersize=12, markeredgewidth=3)
ax_splane.axvline(0, color='black', lw=2) # Ölüm Sınırı (İmajiner Eksen)
ax_splane.axhline(0, color='gray', lw=1, linestyle='--')

# Grafiği sabit tut ki köklerin hareketini algılayabil
ax_splane.set_xlim(-30, 5)
ax_splane.set_ylim(-30, 30)
ax_splane.set_title("S-Düzlemi (Matematiksel Zihin)", fontsize=14)
ax_splane.set_xlabel("Sönümlenme Hızı (Real)")
ax_splane.set_ylabel("Titreşim Frekansı (Imaginary)")
ax_splane.grid(True)

# Bilgi Kutusu
info_text = ax_splane.text(-28, 22, '', fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

# --- SAĞ GRAFİK: ZAMAN YANITI (FİZİKSEL GERÇEKLİK) ---
l_plot, = ax_time.plot(t, y, 'b-', lw=2)
ax_time.axhline(1, color='red', linestyle='--', label='Hedef Açı (1 Radyan)')
ax_time.set_xlim(0, 3)
ax_time.set_ylim(0, 2)
ax_time.set_title("Basamak Yanıtı (Fiziksel Çıktı)", fontsize=14)
ax_time.set_xlabel("Zaman (Saniye)")
ax_time.set_ylabel("Drone Pitch Açısı")
ax_time.legend()
ax_time.grid(True)

# --- KONTROL SÜRGÜLERİ (SLIDERS) ---
ax_Kp = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_Kd = plt.axes([0.15, 0.08, 0.7, 0.03])

# Kp: Suni Yay. Hedefe ne kadar şiddetli çekecek?
slider_Kp = Slider(ax_Kp, 'Kp (Yay/Güç)', 0.01, 10.0, valinit=initial_Kp, color='green')
# Kd: Suni Sürtünme. Hızı nasıl frenleyecek?
slider_Kd = Slider(ax_Kd, 'Kd (Fren/Sönüm)', 0.0, 1.0, valinit=initial_Kd, color='orange')

def update(val):
    Kp = slider_Kp.val
    Kd = slider_Kd.val
    
    poles, t, y, zeta, wn = calc_system(Kp, Kd)
    
    # Köklerin yerini güncelle
    p_plot.set_xdata(np.real(poles))
    p_plot.set_ydata(np.imag(poles))
    
    # Drone'un hareketini güncelle
    l_plot.set_xdata(t)
    l_plot.set_ydata(y)
    
    # Metni güncelle
    info_text.set_text(f"Damping Ratio (Zeta): {zeta:.3f}\n"
                       f"Natural Freq (Wn): {wn:.1f} rad/s\n"
                       f"Kök 1: {poles[0]:.2f}\n"
                       f"Kök 2: {poles[1]:.2f}")
    
    fig.canvas.draw_idle()

slider_Kp.on_changed(update)
slider_Kd.on_changed(update)

plt.show()
