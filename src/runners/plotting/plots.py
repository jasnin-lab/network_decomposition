import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

parameters = [5, 10, 15, 20, 25, 30]

tps = np.array([7, 31, 55, 70, 79, 82])  # True Positives
fps = np.array([25, 133, 335, 677, 844, 883])  # False Positives
fns = np.array([234, 210, 186, 171, 162, 159])  # False Negatives

# Select the index of the parameter you chose to highlight it
chosen_index = 2
chosen_param = parameters[chosen_index]

precision = np.divide(tps, (tps + fps), out=np.zeros_like(tps, dtype=float), where=(tps + fps) != 0)
recall = np.divide(tps, (tps + fns), out=np.zeros_like(tps, dtype=float), where=(tps + fns) != 0)

# F1-Score is the harmonic mean of Precision and Recall
f1_scores = np.divide(2 * (precision * recall), (precision + recall), out=np.zeros_like(precision), where=(precision + recall) != 0)

# ==========================================
# 3. GENERATE PLOTS
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
plt.style.use('seaborn-v0_8-whitegrid') # Gives a clean, academic look

# --- Plot 1: Dual-Line Plot (TP vs FP) ---
ax1.plot(parameters, tps, label='True Positives', color='tab:blue', marker='o', linewidth=2)
ax1.plot(parameters, fps, label='False Positives', color='tab:red', marker='s', linewidth=2)
ax1.axvline(x=chosen_param, color='gray', linestyle='--', label=f'Chosen Param ({chosen_param})')
for x, tp, fp in zip(parameters, tps.tolist(), fps.tolist()):
    ax1.annotate(f'{tp}', (x, tp), textcoords="offset points", xytext=(0,5), ha='center', fontsize=10)
    ax1.annotate(f'{fp}', (x, fp), textcoords="offset points", xytext=(0,5), ha='center', fontsize=10)
ax1.set_title('TP and FP vs. Parameter Value')
ax1.set_xlabel('Parameter Value')
ax1.set_ylabel('Count')
ax1.legend()

# --- Plot 2: ROC-Style Curve (FP vs TP) ---
ax2.plot(fps, tps, color='tab:green', marker='o', linewidth=2)
ax2.scatter(fps[chosen_index], tps[chosen_index], color='red', s=100, zorder=5, label='Chosen Parameter')
# Annotate each point with its parameter value
for i, param in enumerate(parameters):
    ax2.annotate(f' p={param}', (fps[i], tps[i]), textcoords="offset points", xytext=(5,-5), ha='left')
ax2.set_title('True Positives vs. False Positives')
ax2.set_xlabel('False Positives')
ax2.set_ylabel('True Positives')
ax2.legend()

# --- Plot 3: F1-Score Plot ---
ax3.plot(parameters, f1_scores, color='tab:purple', marker='^', linewidth=2)
ax3.axvline(x=chosen_param, color='gray', linestyle='--', label=f'Chosen Param ({chosen_param})')
for x, f1 in zip(parameters, f1_scores):
    ax3.annotate(f'{f1:.4f}', (x, f1), textcoords="offset points", xytext=(0,5), ha='center', fontsize=10)
ax3.set_title('F1-Score vs. Parameter Value')
ax3.set_xlabel('Parameter Value')
ax3.set_ylabel('F1-Score')
ax3.legend()

# Layout adjustment and display
plt.tight_layout()
# plt.show()
plt.savefig("C:/Users/ant/Desktop/Ich_hehe/Studium_Bioinformatik/Bachelor Thesis/Final_Report/Illustrations/param_det.jpg")
