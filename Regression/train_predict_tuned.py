#!/usr/bin/env python3
"""
Seoul Bike Demand Prediction - Tuned LightGBM Regression Pipeline

What this version does:
1. Preprocesses train/test data using the existing feature engineering pipeline.
2. Uses 5-fold cross-validation to select the best LightGBM hyperparameters
   based on validation MSE.
3. Prints mean CV train MSE, CV validation MSE/RMSE/MAE.
4. Plots training and validation MSE against the number of estimators for
   the selected hyperparameters.
5. Fits the final model on all training data.
6. Generates predictions for test.csv and submission.csv.

Important:
- If test.csv does not contain "Rented Bike Count", a true test MSE cannot
  be calculated because the ground-truth target is unavailable.
- In that case, the CV validation MSE is the appropriate estimate of
  generalization/test performance.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


def preprocess_data(df, is_train=True):
    """Apply the existing feature engineering pipeline."""
    df_feat = df.copy()

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

    # Date/time features
    df_feat['Date_dt'] = pd.to_datetime(df_feat['Date'], format='%d/%m/%Y')
    df_feat['Year'] = df_feat['Date_dt'].dt.year
    df_feat['Month'] = df_feat['Date_dt'].dt.month
    df_feat['Day'] = df_feat['Date_dt'].dt.day
    df_feat['DayOfWeek'] = df_feat['Date_dt'].dt.dayofweek
    df_feat['Is_Weekend'] = (df_feat['DayOfWeek'] >= 5).astype(int)

    # Correct season labels from month
    month_to_season = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df_feat['Seasons'] = df_feat['Month'].map(month_to_season)

    # Cyclical features
    df_feat['Hour_sin'] = np.sin(2 * np.pi * df_feat['Hour'] / 24.0)
    df_feat['Hour_cos'] = np.cos(2 * np.pi * df_feat['Hour'] / 24.0)
    df_feat['Month_sin'] = np.sin(2 * np.pi * df_feat['Month'] / 12.0)
    df_feat['Month_cos'] = np.cos(2 * np.pi * df_feat['Month'] / 12.0)
    df_feat['DayOfWeek_sin'] = np.sin(2 * np.pi * df_feat['DayOfWeek'] / 7.0)
    df_feat['DayOfWeek_cos'] = np.cos(2 * np.pi * df_feat['DayOfWeek'] / 7.0)

    # Weather preprocessing
    num_cols = ['Temperature', 'Wind speed', 'Solar Radiation', 'Rainfall']
    for col in num_cols:
        if col in df_feat.columns:
            df_feat[col] = (
                df_feat.groupby(['Month', 'Hour'])[col]
                .transform(lambda x: x.fillna(x.median()))
            )
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())

    df_feat['Is_Raining'] = (df_feat['Rainfall'] > 0).astype(int)
    df_feat['Is_Snowing'] = (df_feat['Snowfall'] > 0).astype(int)
    df_feat['Temp_Humidity_Interaction'] = (
        df_feat['Temperature'] * (df_feat['Humidity'] / 100.0)
    )
    df_feat['Temp_Solar_Interaction'] = (
        df_feat['Temperature'] * df_feat['Solar Radiation']
    )

    # Remove metadata
    drop_cols = ['Date', 'Date_dt', 'Kaggle_ID', 'Record_id']
    drop_cols = [c for c in drop_cols if c in df_feat.columns]
    df_feat = df_feat.drop(columns=drop_cols)

    # One-hot encoding
    df_encoded = pd.get_dummies(
        df_feat,
        columns=['Seasons', 'Holiday', 'Functioning Day'],
        drop_first=True
    )

    return df_encoded


def get_model(params=None):
    """Create a LightGBM regressor."""
    base_params = {
        'n_estimators': 700,
        'learning_rate': 0.03,
        'num_leaves': 40,
        'min_child_samples': 30,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }

    if params:
        base_params.update(params)

    return LGBMRegressor(**base_params)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'train.csv')
    test_path = os.path.join(base_dir, 'test.csv')
    sub_path = os.path.join(base_dir, 'submission.csv')
    plot_path = os.path.join(base_dir, 'training_vs_validation_mse.png')

    print("Starting tuned Seoul Bike Demand regression pipeline...")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Missing train.csv or test.csv in {base_dir}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(
        f"Loaded Train Set: {train_df.shape[0]} records | "
        f"Test Set: {test_df.shape[0]} records"
    )

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------
    print("\nPreprocessing and feature engineering...")
    train_proc = preprocess_data(train_df, is_train=True)
    test_proc = preprocess_data(test_df, is_train=False)

    X_train = train_proc.drop(columns=['Rented Bike Count'])
    y_train = train_proc['Rented Bike Count']
    X_test = test_proc

    X_train, X_test = X_train.align(
        X_test,
        join='left',
        axis=1,
        fill_value=0
    )

    print(f"Feature count: {X_train.shape[1]}")

    # ---------------------------------------------------------
    # CROSS-VALIDATION
    # ---------------------------------------------------------
    # Keep the same 5-fold CV strategy as the original script.
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # MSE is the selection metric.
    # The grid is intentionally moderate so that the search is practical.
    param_grid = {
        'n_estimators': [300, 500, 700, 1000],
        'learning_rate': [0.02, 0.03, 0.05],
        'num_leaves': [25, 40, 60],
        'min_child_samples': [20, 30, 50],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    print("\nSearching for best parameters using 5-fold CV...")
    print("Selection metric: validation MSE")
    print("This may take some time.")

    base_model = LGBMRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_cv_mse = -search.best_score_

    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)

    for key, value in best_params.items():
        print(f"{key:22s}: {value}")

    print(f"\nBest mean CV MSE : {best_cv_mse:.4f}")
    print(f"Best mean CV RMSE: {np.sqrt(best_cv_mse):.4f}")

    # ---------------------------------------------------------
    # DETAILED CV RESULTS FOR BEST MODEL
    # ---------------------------------------------------------
    print("\nEvaluating best model with cross-validation...")

    best_model = get_model(best_params)

    cv_results = cross_validate(
        best_model,
        X_train,
        y_train,
        cv=kf,
        scoring={
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        },
        return_train_score=True,
        n_jobs=-1
    )

    train_mse = -cv_results['train_mse']
    val_mse = -cv_results['test_mse']
    train_mae = -cv_results['train_mae']
    val_mae = -cv_results['test_mae']
    train_r2 = cv_results['train_r2']
    val_r2 = cv_results['test_r2']

    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)

    for i in range(5):
        print(
            f"Fold {i + 1}: "
            f"Train MSE = {train_mse[i]:.2f} | "
            f"Validation MSE = {val_mse[i]:.2f} | "
            f"Validation RMSE = {np.sqrt(val_mse[i]):.2f} | "
            f"Validation R² = {val_r2[i]:.4f}"
        )

    print("\nMean results:")
    print(f"Train MSE        : {np.mean(train_mse):.2f}")
    print(f"Validation MSE   : {np.mean(val_mse):.2f}")
    print(f"Train RMSE       : {np.sqrt(np.mean(train_mse)):.2f}")
    print(f"Validation RMSE  : {np.sqrt(np.mean(val_mse)):.2f}")
    print(f"Train MAE        : {np.mean(train_mae):.2f}")
    print(f"Validation MAE   : {np.mean(val_mae):.2f}")
    print(f"Train R²         : {np.mean(train_r2):.4f}")
    print(f"Validation R²    : {np.mean(val_r2):.4f}")
    print(f"Validation MSE SD: {np.std(val_mse):.2f}")

    # ---------------------------------------------------------
    # TRAINING VS VALIDATION ERROR CURVE
    # ---------------------------------------------------------
    # We keep the best hyperparameters except n_estimators and show
    # how training and CV validation MSE change as trees are added.
    best_n_estimators = best_params['n_estimators']

    curve_estimators = sorted(set(
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
         best_n_estimators]
    ))

    curve_estimators = [
        n for n in curve_estimators
        if n <= max(1000, best_n_estimators)
    ]

    print("\nCalculating training/validation MSE curve...")

    curve_train_mse = []
    curve_val_mse = []

    curve_params = best_params.copy()

    for n_estimators in curve_estimators:
        curve_params['n_estimators'] = n_estimators
        model = get_model(curve_params)

        result = cross_validate(
            model,
            X_train,
            y_train,
            cv=kf,
            scoring='neg_mean_squared_error',
            return_train_score=True,
            n_jobs=-1
        )

        curve_train_mse.append(-np.mean(result['train_score']))
        curve_val_mse.append(-np.mean(result['test_score']))

        print(
            f"n_estimators={n_estimators:4d} | "
            f"Train MSE={curve_train_mse[-1]:.2f} | "
            f"Validation MSE={curve_val_mse[-1]:.2f}"
        )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        curve_estimators,
        curve_train_mse,
        marker='o',
        label='Training MSE'
    )
    plt.plot(
        curve_estimators,
        curve_val_mse,
        marker='o',
        label='Validation MSE'
    )

    plt.axvline(
        best_n_estimators,
        linestyle='--',
        label=f'Best n_estimators = {best_n_estimators}'
    )

    plt.xlabel('Number of estimators')
    plt.ylabel('Mean Squared Error')
    plt.title('Training vs Validation MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print(f"\nSaved error plot to: {plot_path}")

    # ---------------------------------------------------------
    # FINAL MODEL
    # ---------------------------------------------------------
    print("\nTraining final model on the complete training set...")

    final_model = get_model(best_params)
    final_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # TEST ERROR
    # ---------------------------------------------------------
    if 'Rented Bike Count' in test_df.columns:
        # This branch is only possible if a labelled test set is supplied.
        test_preds = np.clip(
            final_model.predict(X_test),
            0,
            None
        )

        y_test = test_df['Rented Bike Count'].values

        test_mse = mean_squared_error(y_test, test_preds)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)

        print("\n" + "=" * 70)
        print("TRUE TEST ERROR")
        print("=" * 70)
        print(f"Test MSE  : {test_mse:.2f}")
        print(f"Test RMSE : {test_rmse:.2f}")
        print(f"Test MAE  : {test_mae:.2f}")
        print(f"Test R²   : {test_r2:.4f}")
    else:
        print("\n" + "=" * 70)
        print("TEST ERROR")
        print("=" * 70)
        print(
            "A true test MSE cannot be calculated because test.csv does not "
            "contain 'Rented Bike Count'."
        )
        print(
            f"Best available generalization estimate "
            f"(5-fold CV validation MSE): {np.mean(val_mse):.2f}"
        )
        print(
            f"Best available generalization estimate "
            f"(5-fold CV validation RMSE): {np.sqrt(np.mean(val_mse)):.2f}"
        )

    # ---------------------------------------------------------
    # PREDICTIONS / SUBMISSION
    # ---------------------------------------------------------
    print("\nGenerating predictions for test.csv...")

    test_preds = np.clip(
        final_model.predict(X_test),
        0,
        None
    )

    test_preds_rounded = np.round(test_preds).astype(int)

    if 'Kaggle_ID' in test_df.columns:
        sub_df = pd.DataFrame({
            'Kaggle_ID': test_df['Kaggle_ID'],
            'Rented Bike Count': test_preds_rounded
        })
    else:
        sub_df = pd.DataFrame({
            'Rented Bike Count': test_preds_rounded
        })

    sub_df.to_csv(sub_path, index=False)

    print(f"Saved submission to: {sub_path}")
    print("\nPrediction summary:")
    print(sub_df.describe().T)

    print("\nDone.")


if __name__ == '__main__':
    main()
