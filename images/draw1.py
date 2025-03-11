import matplotlib.pyplot as plt

# 数据准备
beta_values = [0.5, 1.0, 1.5, 2.0]
# 准确率
accuracy = [0.89, 0.90, 0.91, 0.89]
# 精确率
precision = [0.88, 0.89, 0.92, 0.94]
# 召回率
recall = [0.87, 0.90, 0.91, 0.86]
# F1值
f1 = [0.88, 0.89, 0.92, 0.90]

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(beta_values, accuracy, marker='o', label='准确率')
plt.plot(beta_values, precision, marker='s', label='精确率')
plt.plot(beta_values, recall, marker='^', label='召回率')
plt.plot(beta_values, f1, marker='D', label='F1值')

# 设置坐标轴标签和标题
plt.xlabel('β值')
plt.ylabel('指标值')
plt.title('不同β值下的性能指标对比')

# 设置x轴刻度
plt.xticks(beta_values)

# 显示网格
plt.grid(True)

# 显示图例
plt.legend()
plt.savefig('metric.png')
# 显示图形
plt.show()