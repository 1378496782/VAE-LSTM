import matplotlib.pyplot as plt
import numpy as np

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模拟正常样本和异常样本在 beta=1.5 时的潜在空间数据
np.random.seed(42)
normal_samples_beta_1_5 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
anomaly_samples_beta_1_5 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 50)

# 模拟正常样本和异常样本在 beta=2.0 时的潜在空间数据
# 假设 beta=2.0 时正常样本分布更集中，异常样本有部分混入正常区域
normal_samples_beta_2_0 = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], 100)
anomaly_samples_beta_2_0 = np.vstack([
    np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 40),
    np.random.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], 10)
])

# 创建一个包含两个子图的画布
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 绘制 beta=1.5 时的潜在空间分布
axes[0].scatter(normal_samples_beta_1_5[:, 0], normal_samples_beta_1_5[:, 1], c='blue', label='正常样本', alpha=0.6)
axes[0].scatter(anomaly_samples_beta_1_5[:, 0], anomaly_samples_beta_1_5[:, 1], c='red', label='异常样本', alpha=0.6)
axes[0].set_title('$\mathbf{\\beta = 1.5}$ 潜在空间分布')
axes[0].set_xlabel('$z_1$')
axes[0].set_ylabel('$z_2$')
axes[0].legend()
axes[0].grid(True)

# 绘制 beta=2.0 时的潜在空间分布
axes[1].scatter(normal_samples_beta_2_0[:, 0], normal_samples_beta_2_0[:, 1], c='blue', label='正常样本', alpha=0.6)
axes[1].scatter(anomaly_samples_beta_2_0[:, 0], anomaly_samples_beta_2_0[:, 1], c='red', label='异常样本', alpha=0.6)
axes[1].set_title('$\mathbf{\\beta = 2.0}$ 潜在空间分布')
axes[1].set_xlabel('$z_1$')
axes[1].set_ylabel('$z_2$')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('latent_space.png')
plt.show()