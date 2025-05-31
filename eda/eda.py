import pandas as pd
import polars as ps
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
pd.set_option("display.max_columns", None)

# df = ps.read_csv("../data/creditcard.csv", infer_schema_length=10000, schema_overrides={"Time": ps.Float64})
# df.head()
df_pandas = pd.read_csv("../data/creditcard.csv")

df_pandas.info()
df_pandas.describe()
df_pandas.head()

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

plt.figure(figsize=(25, 25))
sns.heatmap(df_pandas.select_dtypes("number").corr(), cmap="coolwarm", annot=True, fmt=".2f", cbar=True, square=True)
plt.show()
# There are no categorical values to make some visualizations

for col in df_pandas.columns:
    sns.histplot(data=df_pandas, x=col, hue="Class")
    plt.show()

# Steps to look into: Transform some features that are skewed, apply Smote and Scaler to all features

sc = StandardScaler()
df_pandas["Amount"] = sc.fit_transform(df_pandas["Amount"].values.reshape(-1, 1))
df_pandas["Time"] = sc.fit_transform(df_pandas["Time"].values.reshape(-1, 1))
df_pandas.head()

X = df_pandas.drop(columns=("Class"))
y = df_pandas["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)
X_train.head()


sns.histplot(data=y_train)

# Balanced dataframe
# Correlated features with Class = V3, V7, V10, V12, V14, V17
# Correlation Time vs V3, V11, V25 (negative correlated)
# Correlation Amount vs V1, V2, V3, V5, V7, V20
