import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 模型数据
models = ['VAE', 'VAE-LSTM', 'βVAE-LSTM']
accuracy = [0.821, 0.883, 0.91]
precision = [0.856, 0.902, 0.92]
recall = [0.783, 0.865, 0.91]
f1 = [0.818, 0.883, 0.92]

# 设置画布
# 设置分组柱状图参数
width = 0.2
x = np.arange(len(models)) * 1.5  # 扩大模型间距
metrics = ['准确率', '精确率', '召回率', 'F1值']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.figure(figsize=(14, 7))

# 绘制分组柱状图
for i, (metric, values) in enumerate(zip(metrics, [accuracy, precision, recall, f1])):
    offset = width * (i - 1.5)
    bars = plt.bar(x + offset, values, width, label=models, color=colors)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 设置图表样式
plt.xticks(x, models, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('算法模型', fontsize=14)
plt.ylabel('指标值', fontsize=14)
plt.title('不同模型性能指标对比', fontsize=16, pad=20)
plt.grid(axis='y', alpha=0.5)

# 添加图例
handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(models))]
plt.legend(handles, metrics, 
           title='性能指标', 
           fontsize=10, 
           title_fontsize=12,
           loc='upper left', 
           bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig('alg_compare2.png', bbox_inches='tight')
plt.show()