import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(".."))
import cloudpickle
import numpy as np

from utils.transformations import Transformations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def model_train():
    df = pd.read_csv("../data/creditcard.csv")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    df.info()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # smote = SMOTE()
    # X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    std_cols = ["Amount", "Time"]

    std_transformer = Pipeline(steps=[
        ("standard_scaler", StandardScaler())
    ])

    preprocessing = ColumnTransformer(transformers=[
        ("std_transformer", std_transformer, std_cols)
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessing), 
        ("smote", SMOTE()),
        ("model", XGBClassifier(base_score= None,
                                booster= None,
                                callbacks= None,
                                colsample_bylevel= None,
                                colsample_bynode= None,
                                colsample_bytree= None,
                                device= None,
                                early_stopping_rounds= None,
                                enable_categorical= False,
                                eval_metric= "logloss",
                                feature_types= None,
                                gamma= None,
                                grow_policy= None,
                                importance_type= None,
                                interaction_constraints= None,
                                learning_rate= None,
                                max_bin= None,
                                max_cat_threshold= None,
                                max_cat_to_onehot= None,
                                max_delta_step= None,
                                max_depth= 6,
                                max_leaves= None,
                                min_child_weight= None,
                                missing= np.nan,
                                monotone_constraints= None,
                                multi_strategy= None,
                                n_estimators= 100,
                                n_jobs= None,
                                num_parallel_tree= None,
                                objective= "binary:logistic",
                                random_state= None,
                                reg_alpha= None,
                                reg_lambda= None,
                                sampling_method= None,
                                scale_pos_weight= 10,
                                subsample= None,
                                tree_method= None,
                                validate_parameters= None,
                                verbosity= None))
            ])
    
    pipeline.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        cloudpickle.dump(pipeline, f)

    return pipeline

if __name__ == "__is_main":
    pipeline = model_train()