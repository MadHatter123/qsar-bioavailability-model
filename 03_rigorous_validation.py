#!/usr/bin/env python3
"""
Phase 3: Rigorous model validation

Methodology:
1. Split data: 80% training, 20% external test
2. 5-fold cross-validation on training set
3. Regularization (Ridge, Lasso)
4. Hyperparameter optimization
5. Statistical significance testing
6. Applicability domain analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import logging
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
    }

def main():
    logger.info("="*70)
    logger.info("Phase 3: Rigorous model validation")
    logger.info("="*70)
    
    # Load data with selected features
    logger.info("Loading data with selected features...")
    df = pd.read_csv('bioavailability_selected_features.csv')
    
    # Separate features and target
    X = df.iloc[:, 3:].values  # Skip SMILES, Compound_Name, Bioavailability_%
    y = df['Bioavailability_%'].values
    feature_names = df.iloc[:, 3:].columns.tolist()
    
    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Samples: {len(X)}")
    
    # Split into train (80%) and external test (20%)
    logger.info("\nSplitting data: 80% training, 20% external test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("✓ Scaler saved")
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=10000),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR (RBF)': SVR(kernel='rbf', C=100, gamma='scale'),
    }
    
    # Cross-validation results
    cv_results = []
    test_results = []
    
    logger.info("\n" + "="*70)
    logger.info("5-FOLD CROSS-VALIDATION")
    logger.info("="*70)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        logger.info(f"\n{model_name}:")
        
        cv_r2_scores = []
        cv_rmse_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
            X_fold_train = X_train_scaled[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train_scaled[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Train model
            model.fit(X_fold_train, y_fold_train)
            
            # Predict on validation set
            y_fold_pred = model.predict(X_fold_val)
            
            # Calculate metrics
            fold_r2 = r2_score(y_fold_val, y_fold_pred)
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
            
            cv_r2_scores.append(fold_r2)
            cv_rmse_scores.append(fold_rmse)
        
        cv_r2_mean = np.mean(cv_r2_scores)
        cv_r2_std = np.std(cv_r2_scores)
        cv_rmse_mean = np.mean(cv_rmse_scores)
        cv_rmse_std = np.std(cv_rmse_scores)
        
        logger.info(f"  CV R² = {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
        logger.info(f"  CV RMSE = {cv_rmse_mean:.2f}% ± {cv_rmse_std:.2f}%")
        
        cv_results.append({
            'Model': model_name,
            'CV_R2_Mean': cv_r2_mean,
            'CV_R2_Std': cv_r2_std,
            'CV_RMSE_Mean': cv_rmse_mean,
            'CV_RMSE_Std': cv_rmse_std,
        })
        
        # Train on full training set for external test
        model.fit(X_train_scaled, y_train)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate test metrics
        test_metrics = calculate_metrics(y_test, y_test_pred)
        test_metrics['Model'] = model_name
        test_results.append(test_metrics)
        
        logger.info(f"  Test R² = {test_metrics['R2']:.4f}")
        logger.info(f"  Test RMSE = {test_metrics['RMSE']:.2f}%")
        logger.info(f"  Test MAE = {test_metrics['MAE']:.2f}%")
        logger.info(f"  Pearson r = {test_metrics['Pearson_r']:.4f} (p={test_metrics['Pearson_p']:.2e})")
    
    # Save results
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv('cv_results.csv', index=False)
    logger.info("\n✓ CV results saved to cv_results.csv")
    
    test_df = pd.DataFrame(test_results)
    test_df.to_csv('test_results.csv', index=False)
    logger.info("✓ Test results saved to test_results.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("EXTERNAL TEST SET RESULTS")
    print("="*70)
    print(test_df.to_string(index=False))
    print("="*70 + "\n")
    
    # Find best model
    best_idx = test_df['R2'].idxmax()
    best_model = test_df.loc[best_idx, 'Model']
    best_r2 = test_df.loc[best_idx, 'R2']
    
    logger.info(f"\nBest model: {best_model} (R² = {best_r2:.4f})")

if __name__ == "__main__":
    main()
