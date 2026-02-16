#!/usr/bin/env python3
"""
Classification Model - Income Prediction
Predicts whether individuals earn >$50k annually
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_data(data_path, columns_path):
    print("Loading census data...")
    
    with open(columns_path, 'r') as f:
        cols = [line.strip() for line in f if line.strip()]
    
    df = pd.read_csv(data_path, header=None)
    
    if len(cols) == df.shape[1]:
        df.columns = cols
    else:
        df.columns = cols[:df.shape[1]]
    
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    target_col = df.columns[-1]
    df['high_income'] = df[target_col].astype(str).apply(
        lambda x: 1 if '50000+.' in x else 0
    )
    
    df = df.replace(' ?', np.nan)
    df = df.replace('?', np.nan)
    
    return df, target_col


def prepare_features(df, target_col):
    print("\nPreparing features...")
    
    drop_cols = [target_col, 'high_income', 'weight']
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(drop_cols, axis=1)
    y = df['high_income']
    
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if num_cols:
        imp_num = SimpleImputer(strategy='median')
        X[num_cols] = imp_num.fit_transform(X[num_cols])
    
    if cat_cols:
        imp_cat = SimpleImputer(strategy='constant', fill_value='Unknown')
        X[cat_cols] = imp_cat.fit_transform(X[cat_cols])
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y


def train_model(X, y):
    print("\nTraining Random Forest classifier...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    
    if accuracy > 0.95:
        print("High accuracy detected - verify no data leakage")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['<$50k', '>$50k']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(importances.head(10).to_string(index=False))
    
    return clf, importances


def main():
    DATA_FILE = 'census-bureau.data'
    COLS_FILE = 'census-bureau.columns'
    
    try:
        df, target_col = load_data(DATA_FILE, COLS_FILE)
        X, y = prepare_features(df, target_col)
        model, feature_importance = train_model(X, y)
        
        print("\n" + "="*60)
        print("Classification model training complete!")
        print("="*60)
        
    except FileNotFoundError:
        print("\nData files not found. Need:")
        print("  - census-bureau.data")
        print("  - census-bureau.columns")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
