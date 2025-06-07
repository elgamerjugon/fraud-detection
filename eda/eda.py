import pandas as pd
import polars as ps
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

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

# Trying to analyze each column with countplot
df_pandas["Class"].value_counts()

ax = sns.countplot(data=df_pandas, x="Class")
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10)
plt.show()

# Checking distributions on all features
for col in df_pandas.columns:
    sns.histplot(data=df_pandas, x=col, hue="Class", kde=True)
    plt.show()

# Check for distributions of features Time and Amount
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df_pandas['Amount'].values
time_val = df_pandas['Time'].values

sns.histplot(amount_val, ax=ax[0], color='r', kde=True)
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.histplot(time_val, ax=ax[1], color='b', kde=True)
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
plt.show()

# Check for correlations using Heatmap on imbalanced dataframe
plt.figure(figsize=(25, 25))
sns.heatmap(df_pandas.select_dtypes("number").corr(), cmap="coolwarm", annot=True, fmt=".2f", cbar=True, square=True)
plt.show()
# Negative correlations V17, V14, V12 and V10 negative correlation. The lower the values the more likely to be fraud

# There are no categorical values to make some visualizations
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

df_pandas.drop(columns=["Time", "Class", "Amount"]).hist(bins=30, figsize=(20, 15))
plt.suptitle("Distributions of V1-V28 Features")

sns.histplot(data=df_pandas, x="Amount", bins=100, hue="Class", log_scale=True)

sns.histplot(data=pd.concat([X_train, y_train.reset_index(drop=True)], axis=1), x="Amount", bins=100, hue="Class", log_scale=True)
df_pandas.Amount.describe()

cols_to_standarize = ["Amount", "Time"]

standard_transformer = Pipeline(steps=[
    ("standard_scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("standarization", standard_transformer, cols_to_standarize)
], remainder="passthrough")

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", DummyClassifier())
])

param_grid = [
    {
        "model": [DummyClassifier()],
        "model__strategy": ["most_frequent", "stratified"]
    },
    {
        "model": [RandomForestClassifier()],
        "model__n_estimators": [10, 20, 30],
        "model__max_depth": [5, 10]
    },
    {
        "model": [LogisticRegression(max_iter=1000)],
        "model__C": [0.01, 0.1, 1]
    },
    {
        "model": [XGBClassifier(eval_metric="logloss")],
        "model__n_estimators": [100],
        "model__max_depth": [3, 6],
        "model__scale_pos_weight": [5, 10]
    },
    {
        "model": [LGBMClassifier()],
        "model__n_estimators": [100],
        "model__num_leaves": [31, 64],
        "model__class_weight": ["balanced"]
    },
    {
        "model": [CatBoostClassifier(verbose=0)],
        "model__depth": [4, 6],
        "model__iterations": [100],
        "model__learning_rate": [0.1]
    }

]

gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, scoring="roc_auc")
gs.fit(X_train, y_train)

from sklearn.metrics import classification_report, roc_auc_score

y_pred = gs.predict(X_test)
y_proba = gs.predict_proba(X_test)[:, 1]

print("🔍 Best Model:", gs.best_estimator_)
print("✅ Best Params:", gs.best_params_)
print("✅ ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# It would be a good idea to use Randomized GridSearch when using lots of parameters
rs = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, cv=5, n_jobs=-1, scoring="roc_auc")
rs.fit(X_train, y_train)
