import matplotlib.pyplot as plt

# 数据
x = [1e-9, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
y_r1 = [59.317, 54.762, 51.276, 41.846, 43.099, 41.087, 41.548]
y_r2 = [26.190, 22.857, 12.857, 0, 0, 0, 0]
y_rl = [58.175, 54.881, 51.105, 41.731, 43.023, 40.929, 41.609]

# 创建图表并添加数据
plt.figure(figsize=(10,6))
plt.plot(x, y_r1, marker='o', linestyle='-', color='r', label='r1')
plt.plot(x, y_r2, marker='o', linestyle='-', color='g', label='r2')
plt.plot(x, y_rl, marker='o', linestyle='-', color='b', label='rl')

# 设置 x 轴和 y 轴的标签
plt.xlabel('log10(x)')
plt.ylabel('y')

# 设置标题
plt.title('Line Chart')

# 设置 x 轴为对数尺度
plt.xscale('log')

# 启用图例
plt.legend()

# 显示图表
plt.savefig("./figure.png")
