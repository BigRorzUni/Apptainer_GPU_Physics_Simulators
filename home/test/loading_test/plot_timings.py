import matplotlib.pyplot as plt
import numpy as np

# Simulators
labels = ["MuJoCo", "Genesis", "MJX", "Newton"]

# Data arrays (MB per env for each environment size)
values_2048 = [0.21, 18.03, 1.83, 8.04]
values_4096 = [2.22, 19.33, 1.28, 14.96]
values_8192 = [4.20, 22.02, 1.34, 34.57]

x = np.arange(len(labels))  # simulator positions
width = 0.25  # bar width

fig, ax = plt.subplots(figsize=(8,5))

rects1 = ax.bar(x - width, values_2048, width, label="2048 envs")
rects2 = ax.bar(x,         values_4096, width, label="4096 envs")
rects3 = ax.bar(x + width, values_8192, width, label="8192 envs")

# Labels and styling
ax.set_ylabel("Load Time (s)")
ax.set_title("Simulator Loading Time Comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig("plots/loadingspeed_comparison.png")
