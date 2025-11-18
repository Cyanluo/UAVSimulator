import numpy as np
import matplotlib.pyplot as plt

# ==== 修改为你的结果文件路径 ====
results_file = "debug/data_02.npz"

data = np.load(results_file)
time = data["time"]
p = data["p"]                       # 实际位置
desired_p = data["desired_p"]       # 目标位置
ep = data["ep"]                     # 位置误差
ev = data["ev"]                     # 速度误差
er = data["er"]                     # 姿态误差
ew = data["ew"]                     # 角速度误差

colors = {
    "x": "#0072B2",  # 深蓝灰
    "y": "#E69F00",  # 橙黄
    "z": "#000000"  # 黑灰
}

# 创建子图
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

# ---------------------------
# 1. 位置曲线
axes[0].plot(time, p[:,0], color=colors["x"], label='x')
axes[0].plot(time, p[:,1], color=colors["y"], label='y')
axes[0].plot(time, p[:,2], color=colors["z"], label='z')
axes[0].plot(time, desired_p[:,0], '--', color=colors["x"], alpha=0.5, label='x_des')
axes[0].plot(time, desired_p[:,1], '--', color=colors["y"], alpha=0.5, label='y_des')
axes[0].plot(time, desired_p[:,2], '--', color=colors["z"], alpha=0.5, label='z_des')
axes[0].set_title("Position")
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Position [m]")
axes[0].legend()
axes[0].grid(True)

# 2. 位置误差
axes[1].plot(time, ep[:,0], color=colors["x"], label='ex')
axes[1].plot(time, ep[:,1], color=colors["y"], label='ey')
axes[1].plot(time, ep[:,2], color=colors["z"], label='ez')
axes[1].set_title("Position Error")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Error [m]")
axes[1].legend()
axes[1].grid(True)

# 3. 速度误差
axes[2].plot(time, ev[:,0], color=colors["x"], label='evx')
axes[2].plot(time, ev[:,1], color=colors["y"], label='evy')
axes[2].plot(time, ev[:,2], color=colors["z"], label='evz')
axes[2].set_title("Velocity Error")
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Velocity Error [m/s]")
axes[2].legend()
axes[2].grid(True)

# 4. 姿态误差
axes[3].plot(time, er[:,0], color=colors["x"], label='roll error')
axes[3].plot(time, er[:,1], color=colors["y"], label='pitch error')
axes[3].plot(time, er[:,2], color=colors["z"], label='yaw error')
axes[3].set_title("Attitude Error")
axes[3].set_xlabel("Time [s]")
axes[3].set_ylabel("Error [rad]")
axes[3].legend()
axes[3].grid(True)

# 5. 角速度误差
axes[4].plot(time, ew[:,0], color=colors["x"], label='wx error')
axes[4].plot(time, ew[:,1], color=colors["y"], label='wy error')
axes[4].plot(time, ew[:,2], color=colors["z"], label='wz error')
axes[4].set_title("Angular Velocity Error")
axes[4].set_xlabel("Time [s]")
axes[4].set_ylabel("Error [rad/s]")
axes[4].legend()
axes[4].grid(True)

axes[5].axis('off')

plt.tight_layout()
plt.show()
