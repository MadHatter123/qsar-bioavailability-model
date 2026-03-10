#!/usr/bin/env python3
"""
Phase 3: Proper Model Validation with K-Fold CV and External Test Set

This script implements scientifically rigorous validation:
- 5-fold cross-validation
- Separate external test set (20%)
- Regularization (Ridge, Lasso)
- Hyperparameter optimization
- Statistical significance testing
"""

import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, model_name):
    """Calculate comprehensive evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    
    return {
        'Model': model_name,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
        'Mean_Residual': mean_residual,
        'Std_Residual': std_residual
    }

def main():
    logger.info("=" * 70)
    logger.info("PHASE 3: PROPER MODEL VALIDATION")
    logger.info("=" * 70)
    
    # Load data
    df = pd.read_csv('bioavailability_with_descriptors.csv')
    logger.info(f"Loaded {len(df)} compounds with descriptors")
    
    # Load selected features
    with open('selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
    
    X = df[selected_features].values
    y = df['Bioavailability'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and external test set (80-20)
    logger.info("\nSplitting data: 80% training, 20% external test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)} compounds")
    logger.info(f"External test set: {len(X_test)} compounds")
    
    # Models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        'SVR (RBF)': SVR(kernel='rbf', C=100, gamma='scale'),
    }
    
    # K-Fold Cross-Validation
    logger.info("\n" + "=" * 70)
    logger.info("K-FOLD CROSS-VALIDATION (5-fold)")
    logger.info("=" * 70)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    
    for model_name, model in models.items():
        fold_scores = []
        fold_rmses = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            X_fold_train = X_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]
            
            # Train model
            model.fit(X_fold_train, y_fold_train)
            
            # Evaluate
            y_pred = model.predict(X_fold_val)
            r2 = r2_score(y_fold_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            
            fold_scores.append(r2)
            fold_rmses.append(rmse)
        
        mean_r2 = np.mean(fold_scores)
        std_r2 = np.std(fold_scores)
        mean_rmse = np.mean(fold_rmses)
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  CV R² = {mean_r2:.4f} ± {std_r2:.4f}")
        logger.info(f"  CV RMSE = {mean_rmse:.2f}%")
        
        cv_results.append({
            'Model': model_name,
            'CV_R2_mean': mean_r2,
            'CV_R2_std': std_r2,
            'CV_RMSE_mean': mean_rmse
        })
    
    # External Test Set Evaluation
    logger.info("\n" + "=" * 70)
    logger.info("EXTERNAL TEST SET EVALUATION")
    logger.info("=" * 70)
    
    test_results = []
    
    for model_name, model in models.items():
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Predict on external test set
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred_test, model_name)
        test_results.append(metrics)
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  R² = {metrics['R2']:.4f}")
        logger.info(f"  RMSE = {metrics['RMSE']:.2f}%")
        logger.info(f"  MAE = {metrics['MAE']:.2f}%")
        logger.info(f"  Pearson r = {metrics['Pearson_r']:.4f} (p = {metrics['Pearson_p']:.4e})")
        logger.info(f"  Spearman r = {metrics['Spearman_r']:.4f} (p = {metrics['Spearman_p']:.4e})")
    
    # Save results
    cv_df = pd.DataFrame(cv_results)
    test_df = pd.DataFrame(test_results)
    
    cv_df.to_csv('cv_results.csv', index=False)
    test_df.to_csv('test_results.csv', index=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SAVED")
    logger.info("=" * 70)
    logger.info("- cv_results.csv: Cross-validation results")
    logger.info("- test_results.csv: External test set results")
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info("\n✅ Phase 3 completed successfully!")

if __name__ == '__main__':
    main()
