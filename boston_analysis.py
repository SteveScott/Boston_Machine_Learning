from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston['target']
instance_count, attr_count = df.shape
pearson = df.corr(method='pearson')
corr_with_target = pearson.iloc[-1][:-1]
print(corr_with_target)
corr = df.corr()
sns.set_palette("BrBG",7)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr,
            cmap=("BrBG"),
            square=True,
            annot=True,
            linewidths=1,
            ax=ax)
sns.pairplot(df)
sns.jointplot(df["NOX"], df["INDUS"])

print(df.describe())

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    columns = list(df)
    for i in columns:
        print(df[i].describe())

