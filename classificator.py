from typing import List, Tuple, TypeVar
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *


Seria = TypeVar("pandasSeria")
Vecotr = TypeVar("Vectorazer")
def RFCls(key:str, vectorazer:Vecotr, x_train:Seria, y_train:Seria, x_test:Seria, y_test:Seria)->Tuple:
    bow = vectorazer.fit_transform(x_train)
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(bow, y_train)
    pred = rfc.predict(vectorazer.transform(x_test))
    precision_ = precision_score(pred, y_test.to_numpy())
    recall_ = recall_score(pred, y_test.to_numpy())
    f1_score_ = f1_score(pred, y_test.to_numpy())
    return key, precision_, recall_, f1_score_

def RFCls_model(key:str, vectorazer:Vecotr, x_train:Seria, y_train:Seria, x_test:Seria, y_test:Seria)->Tuple:
    bow = vectorazer.fit_transform(x_train)
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(bow, y_train)
    pred = rfc.predict(vectorazer.transform(x_test))
    precision_ = precision_score(pred, y_test.to_numpy())
    recall_ = recall_score(pred, y_test.to_numpy())
    f1_score_ = f1_score(pred, y_test.to_numpy())
    return key, precision_, recall_, f1_score_, rfc