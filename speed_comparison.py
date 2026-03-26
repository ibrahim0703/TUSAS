import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# DOSYA YOLLARI — kendi dizinine göre değiştir
# ─────────────────────────────────────────────
GT_PATH   = "mav0/mocap0/data.csv"          # gerçek veri
PRED_PATH = "estimated_trajectory.csv"    # tahmin

SMOOTH = 5   # yumuşatma penceresi (1 = yok, artırınca daha düzgün)
# ─────────────────────────────────────────────


# ── Ground truth yükle ───────────────────────
gt_raw = pd.read_csv(GT_PATH, comment='#')
gt_raw.columns = gt_raw.columns.str.strip()

# İlk kolon timestamp, sonra x y z
ts_col = gt_raw.columns[0]
x_col  = gt_raw.columns[1]
y_col  = gt_raw.columns[2]
z_col  = gt_raw.columns[3]

gt_raw = gt_raw[[ts_col, x_col, y_col, z_col]].dropna()
gt_raw = gt_raw.sort_values(ts_col).reset_index(drop=True)

# Hız hesapla: ardışık pozisyon farkı / zaman farkı
dt = gt_raw[ts_col].diff() * 1e-9  # ns → s
dx = gt_raw[x_col].diff()
dy = gt_raw[y_col].diff()
dz = gt_raw[z_col].diff()
gt_speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt
gt_df = pd.DataFrame({'ts': gt_raw[ts_col], 'speed': gt_speed}).dropna()
gt_df = gt_df[gt_df['speed'] < 50]  # aykırı değerleri filtrele (>50 m/s)


# ── Tahmin yükle ─────────────────────────────
pred_raw = pd.read_csv(PRED_PATH, comment='#')
pred_raw.columns = pred_raw.columns.str.strip()

ts_col_p     = pred_raw.columns[0]
speed_col    = [c for c in pred_raw.columns if 'speed' in c.lower()]
speed_col    = speed_col[0] if speed_col else pred_raw.columns[4]

pred_df = pred_raw[[ts_col_p, speed_col]].dropna()
pred_df.columns = ['ts', 'speed']
pred_df = pred_df.sort_values('ts').reset_index(drop=True)


# ── Zaman aralığını eşle ─────────────────────
min_ts = max(gt_df['ts'].iloc[0],  pred_df['ts'].iloc[0])
max_ts = min(gt_df['ts'].iloc[-1], pred_df['ts'].iloc[-1])

gt_slice   = gt_df[(gt_df['ts'] >= min_ts)   & (gt_df['ts'] <= max_ts)].reset_index(drop=True)
pred_slice = pred_df[(pred_df['ts'] >= min_ts) & (pred_df['ts'] <= max_ts)].reset_index(drop=True)

# GT timestamp'lerine pred hızını interpole et
pred_interp = np.interp(gt_slice['ts'].values,
                        pred_slice['ts'].values,
                        pred_slice['speed'].values)

frames = np.arange(len(gt_slice))
gt_spd = gt_slice['speed'].values
pr_spd = pred_interp


# ── Yumuşatma ────────────────────────────────
def moving_avg(arr, w):
    if w <= 1:
        return arr
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

gt_sm = moving_avg(gt_spd, SMOOTH)
pr_sm = moving_avg(pr_spd, SMOOTH)
err   = np.abs(gt_sm - pr_sm)


# ── İstatistikler ────────────────────────────
mae     = err.mean()
max_err = err.max()
gt_mean = gt_sm.mean()
pr_mean = pr_sm.mean()

print("=" * 45)
print(f"  GT Ortalama Hız  : {gt_mean:.4f} m/s")
print(f"  Pred Ortalama Hız: {pr_mean:.4f} m/s")
print(f"  MAE              : {mae:.4f} m/s")
print(f"  Maks. Hata       : {max_err:.4f} m/s")
print(f"  Toplam Frame     : {len(frames)}")
print("=" * 45)


# ── Grafik ───────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 9), facecolor='#0a0c10')
gs  = gridspec.GridSpec(3, 1, hspace=0.45,
                        left=0.07, right=0.97, top=0.93, bottom=0.07)

ax1 = fig.add_subplot(gs[0:2])
ax2 = fig.add_subplot(gs[2])

for ax in [ax1, ax2]:
    ax.set_facecolor('#111520')
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e2535')
    ax.tick_params(colors='#4a5568', labelsize=9)
    ax.yaxis.label.set_color('#4a5568')
    ax.xaxis.label.set_color('#4a5568')
    ax.grid(color='#1e2535', linewidth=0.5, linestyle='--')

# — Ana grafik —
ax1.plot(frames, gt_sm,  color='#00e5ff', linewidth=1.4, label='Ground Truth', alpha=0.9)
ax1.plot(frames, pr_sm,  color='#ff4d6d', linewidth=1.4, label='Tahmin',       alpha=0.9)
ax1.fill_between(frames, gt_sm, pr_sm, alpha=0.12, color='#ffd166', label='Fark alanı')
ax1.set_ylabel('Hız (m/s)')
ax1.set_title('HIZ KARŞILAŞTIRMA — Ground Truth vs Tahmin',
              color='#c8d0e0', fontsize=12, pad=10, loc='left')

legend = ax1.legend(loc='upper right', framealpha=0.3,
                    facecolor='#111520', edgecolor='#1e2535',
                    labelcolor='#c8d0e0', fontsize=9)

# İstatistik kutusu
stats_txt = (f"GT Ort: {gt_mean:.3f} m/s\n"
             f"Pred Ort: {pr_mean:.3f} m/s\n"
             f"MAE: {mae:.4f} m/s\n"
             f"Maks Hata: {max_err:.4f} m/s")
ax1.text(0.01, 0.97, stats_txt, transform=ax1.transAxes,
         va='top', ha='left', fontsize=8.5,
         color='#c8d0e0', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#0a0c10',
                   edgecolor='#1e2535', alpha=0.9))

# — Hata grafiği —
cmap_colors = plt.cm.YlOrRd(err / (max_err + 1e-9))
ax2.bar(frames, err, color=cmap_colors, width=1.0, linewidth=0)
ax2.set_xlabel('Frame')
ax2.set_ylabel('|Hata| (m/s)')
ax2.set_title('MUTLAK HIZ HATASI', color='#c8d0e0', fontsize=10, pad=8, loc='left')

# Ortalama hata çizgisi
ax2.axhline(mae, color='#ffd166', linewidth=1.2, linestyle='--',
            label=f'Ort. Hata: {mae:.4f} m/s')
ax2.legend(loc='upper right', framealpha=0.3,
           facecolor='#111520', edgecolor='#1e2535',
           labelcolor='#c8d0e0', fontsize=8)

plt.savefig('speed_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0a0c10')
print("Grafik kaydedildi: speed_comparison.png")
# plt.show()
