import pandas as pd
from fastapi import FastAPI
import cloudpickle
import sys
import os
sys.path.append(os.path.abspath(".."))

from pydantic import BaseModel
from utils.transformations import Transformations

app = FastAPI()

with open("model.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)

class Transactions(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Time: float
    Amount: float

@app.post("/predict")

def predict(transactions: Transactions):
    X_train = pd.read_csv("../data/creditcard.csv")
    X_train = X_train.drop(columns=["Class"])
    fe = Transformations()
    fe.fit(X_train)
    json = pd.DataFrame([transactions.dict()])
    json = fe.transform(json)
    prediction = pipeline.predict(json)[0]

    return {"Is Fraud?": int(prediction)}
    