import matplotlib.pyplot as plt
import numpy as np

# Generate x and y values
x = np.arange(50, 751, 50)
cos_similarity1 = np.array([0.9997, 0.9992, 0.9987, 0.9976, 0.9969, 0.9956, 0.9953, 0.9938, 0.9906, 0.9886, 0.9980, 0.9856, 0.9831, 0.981, 0.978]) 
# cos_similarity2 = np.array([0.9335, 0.9801, 0.9619, 0.9837, 0.9299, 0.9813, 0.9884, 0.9832,  0.9649, 0.9745, 0.9809, 0.9792, 0.9816, 0.9713, 0.9422])
cos_similarity2 = np.array([0.8475, 0.9365, 0.8611, 0.8562, 0.8524, 0.9033, 0.8534, 0.9212, 0.8411, 0.9672, 0.8375, 0.8370, 0.8353, 0.8348, 0.9452]
)
# cos_similarity3 = np.array([0.6692, 0.9271, 0.7038, 0.9098, 0.7182, 0.9060, 0.8273, 0.9077,  0.8573, 0.9263, 0.7147, 0.6637, 0.7944, 0.7213, 0.7808])
cos_similarity3 = np.array([0.7686, 0.8382, 0.7931, 0.7562, 0.7536, 0.7864, 0.8093, 0.7488, 0.7983, 0.8473, 0.7407, 0.7371, 0.7278, 0.7226, 0.8388]
)

rl_values = np.random.rand(15) * 100  # 15 random values between 0 and 1
r1_values = np.random.rand(15) * 100  # 15 random values between 0 and 100
r2_values = np.random.rand(15) * 100  # 15 random values between 0 and 100

# Plot the values
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Cos similarity and R-L performance on the primary y-axis
ax1.plot(x, cos_similarity1, marker='o', label='Cos Similarity (B=1)', color='blue')
ax1.plot(x, cos_similarity2, marker='p', label='Cos Similarity (B=2)', color='green')
ax1.plot(x, cos_similarity3, marker='d', label='Cos Similarity (B=4)', color='red')

# Set y-axis limits for ax1
ax1.set_ylim([0.6, 1])
ax1.set_ylabel('Cosine Similarity (0.6 to 1)', color='black', fontsize=18)
ax1.set_xlabel("d': Recovered Dimension", fontsize=16)
ax1.tick_params(axis='y', labelcolor='black')
ax1.tick_params(axis='both', labelsize=16)

# Create a secondary y-axis for R-1 and R-2 values
# ax2 = ax1.twinx()
# ax2.plot(x, r1_values, marker='^', label='R-1', color='green')
# ax2.plot(x, r2_values, marker='d', label='R-2', color='purple')
# ax2.plot(x, rl_values, marker='s', label='R-L', color='red')

# Set y-axis limits for ax2
# ax2.set_ylim([0, 100])
# ax2.set_ylabel('Value (0 to 100)', color='black')
# ax2.tick_params(axis='y', labelcolor='black')

# Title, labels, and legend
# plt.title('Comparison of Recovered Cos Similarity, R-L, R-1, and R-2 over d\'')
fig.tight_layout()
handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1, labels1, loc='lower right', fontsize=16)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save the plot to a file (optional)
# plt.savefig('line_plot.png', dpi=300)

# Show the plot
plt.savefig('line_plot.pdf')
