import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
matplotlib.rcParams['axes.unicode_minus'] = False

# 模型数据
models = ['VAE', 'VAE-LSTM', 'βVAE-LSTM']
accuracy = [0.821, 0.883, 0.91]
precision = [0.856, 0.902, 0.92]
recall = [0.783, 0.865, 0.91]
f1 = [0.818, 0.883, 0.92]

# 设置画布
plt.figure(figsize=(12, 6))
x = np.arange(len(models))  # 模型位置
width = 0.2  # 柱子宽度

# 绘制不同指标柱子
plt.bar(x - 1.5*width, accuracy, width, label='准确率')
plt.bar(x - 0.5*width, precision, width, label='精确率')
plt.bar(x + 0.5*width, recall, width, label='召回率')
plt.bar(x + 1.5*width, f1, width, label='F1值')

# 添加细节
plt.xticks(x, models)
plt.xlabel('模型')
plt.ylabel('指标值')
plt.title('不同模型性能指标对比')
plt.legend()

# 添加数据标签
for i, model in enumerate(models):
    plt.text(i - 1.5*width, accuracy[i], f'{accuracy[i]:.3f}', ha='center', va='bottom')
    plt.text(i - 0.5*width, precision[i], f'{precision[i]:.3f}', ha='center', va='bottom')
    plt.text(i + 0.5*width, recall[i], f'{recall[i]:.3f}', ha='center', va='bottom')
    plt.text(i + 1.5*width, f1[i], f'{f1[i]:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('alg_compare.png')
plt.show()