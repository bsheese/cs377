import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

datasets = [
    {"name": "Heart Hungarian", "id": 1498}
]

for ds in datasets:
    try:
        data = fetch_openml(data_id=ds['id'], as_frame=True, parser='auto')
        X = data.data
        y = data.target
        le = LabelEncoder()
        y = le.fit_transform(y)
        for col in X.select_dtypes(['object', 'category']).columns:
            X[col] = X[col].astype('category')
        unique, counts = np.unique(y, return_counts=True)
        labels = le.inverse_transform(unique)
        dist_str = ", ".join([f"{k}: {v}" for k, v in dict(zip(labels, counts)).items()])
        model = XGBClassifier(enable_categorical=True, n_estimators=100, random_state=42)
        cv_results = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'f1_macro'])
        acc = np.mean(cv_results['test_accuracy'])
        f1 = np.mean(cv_results['test_f1_macro'])
        print(f"{ds['name']} (ID: {ds['id']}) - Acc: {acc:.4f}, F1: {f1:.4f}, Dist: {dist_str}")
    except Exception as e:
        print(f"Error {ds['name']}: {e}")
