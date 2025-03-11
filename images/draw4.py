import matplotlib.pyplot as plt
import numpy as np

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模拟不同β值下的潜在空间数据
np.random.seed(42)

# β=0.5（解纠缠不足）
normal_beta05 = np.random.multivariate_normal([0, 0], [[2, 0.5], [0.5, 2]], 100)
anomaly_beta05 = np.random.multivariate_normal([3, 3], [[2, 0.5], [0.5, 2]], 50)

# β=1.0（标准VAE）
normal_beta10 = np.random.multivariate_normal([0, 0], [[1.5, 0], [0, 1.5]], 100)
anomaly_beta10 = np.random.multivariate_normal([4, 4], [[1.5, 0], [0, 1.5]], 50)

# β=1.5（最优解）
normal_beta15 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
anomaly_beta15 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 50)

# β=2.0（过约束）
normal_beta20 = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 100)
anomaly_beta20 = np.vstack([
    np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 40),
    np.random.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], 10)
])

# 创建画布
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
beta_values = [0.5, 1.0, 1.5, 2.0]
plots = [
    (axes[0, 0], normal_beta05, anomaly_beta05),
    (axes[0, 1], normal_beta10, anomaly_beta10),
    (axes[1, 0], normal_beta15, anomaly_beta15),
    (axes[1, 1], normal_beta20, anomaly_beta20)
]

# 绘制子图
for i, (ax, normal, anomaly) in enumerate(plots):
    ax.scatter(normal[:, 0], normal[:, 1], c='blue', label='正常样本', alpha=0.6)
    ax.scatter(anomaly[:, 0], anomaly[:, 1], c='red', label='异常样本', alpha=0.6)
    ax.set_title(f'β = {beta_values[i]}', fontsize=12)
    ax.set_xlabel('$z_1$')
    ax.set_ylabel('$z_2$')
    ax.legend()
    ax.grid(True)

# 突出β=1.5的最优效果
axes[1, 0].set_title('β = 1.5（最优解）', fontsize=12, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('all_latent_space.png')
plt.show()