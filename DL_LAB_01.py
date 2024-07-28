#======= TASK 1 =======

# == 1 ==
import numpy as np
import matplotlib.pyplot as plt

arr1 = np.random.rand(4,4)

# == 2 ==

arr2 = np.random.rand(100000,1)

# print(arr1)
# print(arr2)

plt.hist(np.random.rand(100000), density=True, bins=100, histtype="step", color="blue", label="rand")
plt.axis([-2.5, 2.5, 0, 1.1])
plt.legend(loc = "upper left")
plt.title("Random distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# == 3 ==

fig = plt.figure()
ax = plt.axes(projection ='3d')

x = -5
y = 5
z = x**2 + y**2
ax.plot3D(x, y, z, 'green')
ax.set_title('3D line plot')
plt.show()

# == 4 ==

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
!wget -q https://elitedatascience.com/wp-content/uploads/2022/07/Pokemon.csv
# Read dataset

df = pd.read_csv('Pokemon.csv', index_col=0, encoding='latin')
df.head()

features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
df_filtered = df[features]

pearson_corr = df_filtered.corr(method='pearson')

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Pearson Correlation Matrix')

plt.tight_layout()
plt.show()

spearman_corr = df_filtered.corr(method='spearman')

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 2)
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Spearman Correlation Matrix')

plt.tight_layout()
plt.show()

#======= TASK 2 =======

# == 1 ==
import pandas as pd
df = pd.read_csv('Au_nanoparticle_dataset.csv')
df.head()

filtered_df = df[['N_total', 'N_bulk', 'N_surface', 'R_avg']]
filtered_df.head()

# == 2 ==

filtered_df.head(20)

# == 3 ==

mean_values = filtered_df.mean()
std_values = filtered_df.std()
quartiles = filtered_df.quantile([0.25, 0.5, 0.75])

print("Mean values for each feature:")
print(mean_values)

print("\nStandard deviation for each feature:")
print(std_values)

print("\nQuartile values for each feature:")
print(quartiles)

# == 4 ==

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Plot histogram for 'N_total'
axes[0].hist(filtered_df['N_total'], bins=20, color='skyblue', edgecolor='black')
axes[0].set_title('Histogram of N_total')
axes[0].set_xlabel('N_total')
axes[0].set_ylabel('Frequency')

# Plot histogram for 'N_bulk'
axes[1].hist(filtered_df['N_bulk'], bins=20, color='salmon', edgecolor='black')
axes[1].set_title('Histogram of N_bulk')
axes[1].set_xlabel('N_bulk')
axes[1].set_ylabel('Frequency')

# Plot histogram for 'N_surface'
axes[2].hist(filtered_df['N_surface'], bins=20, color='lightgreen', edgecolor='black')
axes[2].set_title('Histogram of N_surface')
axes[2].set_xlabel('N_surface')
axes[2].set_ylabel('Frequency')

# Plot histogram for 'R_avg'
axes[3].hist(filtered_df['R_avg'], bins=20, color='gold', edgecolor='black')
axes[3].set_title('Histogram of R_avg')
axes[3].set_xlabel('R_avg')
axes[3].set_ylabel('Frequency')

plt.tight_layout()

plt.show()

# == 5 ==

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(filtered_df, diag_kind='hist', plot_kws={'alpha':0.7, 'edgecolor': 'k'})
plt.show()

# == 6 ==

new_df = df[['N_total', 'N_bulk', 'N_surface', 'R_avg']]

g = sns.PairGrid(new_df)

g.map_upper(sns.histplot, kde=False)

g.map_diag(sns.histplot, kde=True)

g.map_lower(sns.kdeplot, cmap='Blues_d')

plt.tight_layout()

plt.show()