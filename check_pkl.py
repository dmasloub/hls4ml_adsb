import pickle

file_path = '/home/david/Bachelor/hls4ml_adsb/checkpoints/optimizer_results.pkl'  


with open(file_path, 'rb') as file:
    data = pickle.load(file)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(data)


print("Basic Statistics:")
print(df.describe())


top_n = 10
print(f"\nTop {top_n} Configurations:")
print(df.nlargest(top_n, 'score')[['bits', 'integer_bits', 'alpha', 'pruning_percent',
                                   'standard_q_threshold', 'accuracy', 'score']])


sns.set(style="whitegrid")

# Plot 1: Accuracy vs Average Normalized Resource Usage
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='average_normalized_resource_usage', y='accuracy', hue='score', palette='viridis', size='score', sizes=(50, 200))
plt.title('Accuracy vs Average Normalized Resource Usage')
plt.xlabel('Average Normalized Resource Usage')
plt.ylabel('Accuracy')
plt.legend(title='Score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot 2: Resource Utilization vs Accuracy
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='lut_utilization_pct', y='accuracy', hue='score', palette='coolwarm', size='score', sizes=(50, 200))
plt.title('Accuracy vs LUT Utilization (%)')
plt.xlabel('LUT Utilization (%)')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='dsp_utilization_pct', y='accuracy', hue='score', palette='coolwarm', size='score', sizes=(50, 200))
plt.title('Accuracy vs DSP Utilization (%)')
plt.xlabel('DSP Utilization (%)')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='ff_utilization_pct', y='accuracy', hue='score', palette='coolwarm', size='score', sizes=(50, 200))
plt.title('Accuracy vs FF Utilization (%)')
plt.xlabel('FF Utilization (%)')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='pruning_percent', y='accuracy', hue='score', palette='coolwarm', size='score', sizes=(50, 200))
plt.title('Accuracy vs Pruning Percent')
plt.xlabel('Pruning Percent')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# Plot 3: Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Plot 4: Top Configurations Bar Chart
top_configs = df.nlargest(top_n, 'score').sort_values('score', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(data=top_configs, x=range(1, top_n+1), y='score', palette='magma')
plt.title(f'Top {top_n} Configurations by Score')
plt.xlabel('Configuration Rank')
plt.ylabel('Score')
plt.tight_layout()
plt.show()

# Plot 5: Parallel Coordinates Plot
from pandas.plotting import parallel_coordinates

df_parallel = df.copy()
df_parallel['score_bin'] = pd.qcut(df_parallel['score'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

plt.figure(figsize=(15, 10))
parallel_coordinates(df_parallel[['bits', 'integer_bits', 'alpha', 'pruning_percent',
                                 'standard_q_threshold', 'accuracy', 'score_bin']], 'score_bin', colormap='viridis')
plt.title('Parallel Coordinates Plot')
plt.xlabel('Features')
plt.ylabel('Values')
plt.legend(title='Score Bin')
plt.tight_layout()
plt.show()

# Plot 6: Pairplot for Key Metrics
sns.pairplot(df, vars=['accuracy', 'score', 'average_normalized_resource_usage',
                       'lut_utilization_pct', 'dsp_utilization_pct', 'ff_utilization_pct'],
             hue='score', palette='viridis', diag_kind='kde')
plt.suptitle('Pairplot of Key Metrics', y=1.02)
plt.show()
