import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取结果数据
results_df = pd.read_csv('enhancement_results.csv')

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Original Image MOS', y='Enhanced Image MOS',
                hue='Predicted Output Value', palette='viridis', size='Predicted Output Value', sizes=(20, 200))
plt.title('Scatter Plot of Original vs Enhanced Image MOS')
plt.xlabel('Original Image MOS')
plt.ylabel('Enhanced Image MOS')

# 添加 y=x 线
lims = [
    np.min([plt.xlim(), plt.ylim()]),  # 坐标轴的最小值
    np.max([plt.xlim(), plt.ylim()]),  # 坐标轴的最大值
]
x = np.linspace(lims[0], lims[1], 100)
plt.plot(x, x, color='red', linestyle='--', label='y=x')  # 绘制 y=x 线
plt.legend(title='Predicted Output Value')

plt.grid()
plt.savefig('scatter_plot.png')  # 保存为图片
plt.show()

# 绘制箱型图
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df[['Original Image MOS', 'Enhanced Image MOS']])
plt.title('Box Plot of Original and Enhanced Image MOS')
plt.ylabel('MOS Value')
plt.xticks(ticks=[0, 1], labels=['Original Image', 'Enhanced Image'])
plt.grid()
plt.savefig('box_plot.png')  # 保存为图片
plt.show()

# 绘制热图
plt.figure(figsize=(10, 6))
heatmap_data = results_df[['Predicted Output Value', 'Original Image MOS', 'Enhanced Image MOS']]
heatmap_data_corr = heatmap_data.corr()
sns.heatmap(heatmap_data_corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('Heatmap of Correlation between Features')
plt.savefig('heatmap.png')  # 保存为图片
plt.show()
