import pandas as pd
import polars as ps
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn import SMOTE
pd.set_option("display.max_columns", None)

# df = ps.read_csv("../data/creditcard.csv", infer_schema_length=10000, schema_overrides={"Time": ps.Float64})
# df.head()
df_pandas = pd.read_csv("../data/creditcard.csv")

df_pandas.info()
df_pandas.describe()

# print(df.schema)
# df.describe
# df.isna().sum()

# Trying to analyze each column
df_pandas["Class"].value_counts()

ax = sns.countplot(data=df_pandas, x="Class")
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10)
plt.show()

# There are no categorical values to make some visualizations

for col in df_pandas.columns:
    sns.histplot(data=df_pandas, x=col, hue="Class")
    plt.show()

# Steps to look into: Transform some features that are skewed, apply Smote and Scaler to all features
