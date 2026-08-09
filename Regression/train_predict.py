#!/usr/bin/env python3
"""
Seoul Bike Demand Prediction - Regression Training and Prediction Pipeline
This script trains a LightGBM Regressor on train.csv, evaluates it using 5-Fold
Cross-Validation, and generates predictions on test.csv in the format of sample_submission.csv.

Data Preprocessing Steps:
1. Column Name Standardisation (aligning Kaggle column names).
2. Date Parsing & Temporal Feature Extraction (Year, Month, Day, DayOfWeek, Is_Weekend).
3. Seasons Typos Correction (mapping corrupted seasons string labels based on date months).
4. Missing Value Imputation (grouped median imputation by Month and Hour for numerical features).
5. Cyclical Encodings (sine and cosine transforms for Hour, Month, DayOfWeek).
6. Weather Interactions (Is_Raining, Is_Snowing indicators, and temperature-humidity/solar interactions).
7. Drop metadata columns (Kaggle_ID, Record_id, Date).
8. One-hot encoding of categorical variables (Seasons, Holiday, Functioning Day).
"""

import os
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def preprocess_data(df, is_train=True):
    """
    Applies the standardized feature engineering pipeline to the raw dataset.
    """
    df_feat = df.copy()
    
    # Standardize column names if Kaggle names are used
    col_rename = {
        'Temperature(°C)': 'Temperature',
        'Humidity(%)': 'Humidity',
        'Wind speed (m/s)': 'Wind speed',
        'Visibility (10m)': 'Visibility',
        'Dew point temperature(°C)': 'Dew point temperature',
        'Solar Radiation (MJ/m2)': 'Solar Radiation',
        'Rainfall(mm)': 'Rainfall',
        'Snowfall (cm)': 'Snowfall'
    }
    df_feat = df_feat.rename(columns=col_rename)
    
    # 1. Date Parsing & Temporal Feature Extraction
    df_feat['Date_dt'] = pd.to_datetime(df_feat['Date'], format='%d/%m/%Y')
    df_feat['Year'] = df_feat['Date_dt'].dt.year
    df_feat['Month'] = df_feat['Date_dt'].dt.month
    df_feat['Day'] = df_feat['Date_dt'].dt.day
    df_feat['DayOfWeek'] = df_feat['Date_dt'].dt.dayofweek
    df_feat['Is_Weekend'] = (df_feat['DayOfWeek'] >= 5).astype(int)
    
    # 2. Season Typos Correction
    # Korean Seasons based on Month: Winter (12,1,2), Spring (3,4,5), Summer (6,7,8), Autumn (9,10,11)
    month_to_season = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df_feat['Seasons'] = df_feat['Month'].map(month_to_season)
    
    # 3. Cyclical Sine/Cosine Encodings
    df_feat['Hour_sin'] = np.sin(2 * np.pi * df_feat['Hour'] / 24.0)
    df_feat['Hour_cos'] = np.cos(2 * np.pi * df_feat['Hour'] / 24.0)
    df_feat['Month_sin'] = np.sin(2 * np.pi * df_feat['Month'] / 12.0)
    df_feat['Month_cos'] = np.cos(2 * np.pi * df_feat['Month'] / 12.0)
    df_feat['DayOfWeek_sin'] = np.sin(2 * np.pi * df_feat['DayOfWeek'] / 7.0)
    df_feat['DayOfWeek_cos'] = np.cos(2 * np.pi * df_feat['DayOfWeek'] / 7.0)
    
    # 4. Weather Indicators & Interactions
    # Impute missing values for interactions using median values grouped by Month/Hour
    num_cols = ['Temperature', 'Wind speed', 'Solar Radiation', 'Rainfall']
    for col in num_cols:
        if col in df_feat.columns:
            df_feat[col] = df_feat.groupby(['Month', 'Hour'])[col].transform(lambda x: x.fillna(x.median()))
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())
            
    df_feat['Is_Raining'] = (df_feat['Rainfall'] > 0).astype(int)
    df_feat['Is_Snowing'] = (df_feat['Snowfall'] > 0).astype(int)
    df_feat['Temp_Humidity_Interaction'] = df_feat['Temperature'] * (df_feat['Humidity'] / 100.0)
    df_feat['Temp_Solar_Interaction'] = df_feat['Temperature'] * df_feat['Solar Radiation']
    
    # 5. Drop non-predictive columns
    drop_cols = ['Date', 'Date_dt', 'Kaggle_ID', 'Record_id']
    drop_cols = [c for c in drop_cols if c in df_feat.columns]
    df_feat = df_feat.drop(columns=drop_cols)
    
    # 6. One-Hot Encoding Categoricals
    df_encoded = pd.get_dummies(df_feat, columns=['Seasons', 'Holiday', 'Functioning Day'], drop_first=True)
    
    return df_encoded

def main():
    # File Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'train.csv')
    test_path = os.path.join(base_dir, 'test.csv')
    sub_path = os.path.join(base_dir, 'submission.csv')
    
    print("🚀 Starting Seoul Bike Demand Prediction Regression Pipeline...")
    
    # Load Datasets
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing train.csv or test.csv in {base_dir}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"📊 Loaded Train Set: {train_df.shape[0]} records | Test Set: {test_df.shape[0]} records")
    
    # Preprocessing
    print("🧹 Preprocessing and engineering features...")
    train_proc = preprocess_data(train_df)
    test_proc = preprocess_data(test_df)
    
    # Split features and target
    X_train = train_proc.drop(columns=['Rented Bike Count'])
    y_train = train_proc['Rented Bike Count']
    X_test = test_proc
    
    # Align columns between train and test
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    print(f"✨ Feature Shape: {X_train.shape[1]} engineered features")
    
    # 5-Fold Cross Validation to audit model generalization performance
    print("\n🔍 Running 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = []
    cv_rmse = []
    cv_mae = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        # Best LightGBM Hyperparameters from optimization
        fold_model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=90,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        fold_model.fit(X_tr, y_tr)
        
        # Clipped predictions (demand count cannot be negative)
        val_preds = np.clip(fold_model.predict(X_val), 0, None)
        
        r2 = r2_score(y_val, val_preds)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)
        
        cv_r2.append(r2)
        cv_rmse.append(rmse)
        cv_mae.append(mae)
        print(f"   Fold {fold}: R² = {r2:.4f} | RMSE = {rmse:.2f} | MAE = {mae:.2f}")
        
    print(f"⭐ 5-Fold Mean Results: R² = {np.mean(cv_r2):.4f} +/- {np.std(cv_r2):.4f} | RMSE = {np.mean(cv_rmse):.2f} | MAE = {np.mean(cv_mae):.2f}")
    
    # Fit Final Model on the entire Training Set
    print("\n🏋️ Training final model on complete training set...")
    final_model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=90,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    final_model.fit(X_train, y_train)
    
    # Predict on test set
    print("🔮 Generating predictions on test set...")
    test_preds = final_model.predict(X_test)
    test_preds_clipped = np.clip(test_preds, 0, None)
    test_preds_rounded = np.round(test_preds_clipped).astype(int)
    
    # Construct final submission DataFrame
    sub_df = pd.DataFrame({
        'Kaggle_ID': test_df['Kaggle_ID'],
        'Rented Bike Count': test_preds_rounded
    })
    
    # Verify shape and format
    assert sub_df.shape == (2628, 2), f"Error: Submission shape {sub_df.shape} is incorrect!"
    assert list(sub_df.columns) == ['Kaggle_ID', 'Rented Bike Count'], "Error: Submission columns are incorrect!"
    
    sub_df.to_csv(sub_path, index=False)
    print(f"✅ Saved final predictions to: {sub_path}")
    print(f"📊 Summary statistics of predictions:\n{sub_df.describe().T}")
    print("\nDone!")

if __name__ == '__main__':
    main()
