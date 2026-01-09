import matplotlib.pyplot as plt
import numpy as np
import math
import time
# 1. SIMULATION PARAMETERS
SPEED = 150
DURATION = 10
DT = 0.5 # delta time
HEADING_ANGLE = 30
ANIMATION_DELAY = 0.1 # HOW LONG WAIT FOR VISUALIZATION

# 2. PARAMETER CALCULATION AND ARRANGEMENT
theta_rad = math.radians(HEADING_ANGLE) # convert degree to radian
v_y = SPEED*math.cos(theta_rad)
v_x = SPEED*math.sin(theta_rad)
print(f"[SYSTEM] Simulation starting...")
print(f"[INFO] Velocity vectors -> Vx:{v_x:.2f}m/s , Vy:{v_y:.2f}m/s")

# 3. WINDOW SETTINGS

fig = plt.figure(figsize =(8,8))
plt.title(f"Aircraft simulation (Speed:{SPEED} m/s, Headings:{HEADING_ANGLE})")
plt.xlabel("Distance X (meters)")
plt.ylabel("Distance Y (meters)")
plt.grid(True,linestyle="--",alpha=0.6)

# fix grid limits to prevent auto-scaling
x,y = 0.0,0.0 # initial position
x_history, y_history = [x],[y]  # list to store trajectory trace (yörünge izi)

# ro = red dot aircraft b-- = blue dash line trail iz
plot_aircraft, = plt.plot(x,y,"ro",markersize = 10,label = "Aircraft")
plot_trail, = plt.plot(x,y,"b--",linewidth = 1,label = "Trajectory")
plt.legend()

plt.ion()
plt.show() # show initial window

# 4. SIMULATION LOOP (CONVERTING DISCRETE TO CONT CASE)
time_steps =np.arange(DT,DURATION+DT,DT)
for t in time_steps: # at each step update x and y coordinate of plane
    x += v_x * DT
    y += v_y * DT
    #record new step coordinate 
    x_history.append(x)
    y_history.append(y)

    # update plot data
    plot_aircraft.set_data([x],[y])
    plot_trail.set_data(x_history,y_history)
    plt.title(f"Time:{t:.1f}s  / Pos:({x:.1f},{y:.1f}")
    # to draw every frame
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(ANIMATION_DELAY)


plt.ioff()
print("[SYSTEM] Simulation Completed.")
plt.show()











