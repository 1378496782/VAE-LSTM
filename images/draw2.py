import matplotlib.pyplot as plt

# 数据准备
beta_values = [0.5, 1.0, 1.5, 2.0]
# 重构损失
recon_loss = [0.082, 0.091, 0.105, 0.120]
# KL 散度
kl_divergence = [0.125, 0.156, 0.182, 0.205]

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制重构损失折线图
color = 'tab:orange'
ax1.set_xlabel('β值')
ax1.set_ylabel('重构损失', color=color)
ax1.plot(beta_values, recon_loss, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个 y 轴用于绘制 KL 散度
ax2 = ax1.twinx()

# 绘制 KL 散度折线图
color = 'tab:blue'
ax2.set_ylabel('KL 散度', color=color)
ax2.plot(beta_values, kl_divergence, color=color, marker='s')
ax2.tick_params(axis='y', labelcolor=color)

# 设置 x 轴刻度
plt.xticks(beta_values)

# 设置标题
plt.title('β值对重构损失与 KL 散度的影响')

# 显示网格
ax1.grid(True)

plt.savefig('kl_loss.png')
# 显示图形
plt.show()