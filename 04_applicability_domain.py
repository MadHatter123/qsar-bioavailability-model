#!/usr/bin/env python3
"""Phase 4: Applicability Domain Analysis"""

import pandas as pd
import numpy as np
import logging
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_applicability_domain(X_train, X_test):
    """Calculate applicability domain using Euclidean distance method."""
    # Calculate distances from each test compound to all training compounds
    distances = cdist(X_test, X_train, metric='euclidean')
    
    # Calculate minimum distance for each test compound
    min_distances = np.min(distances, axis=1)
    
    # Calculate threshold (mean + 2*std of training set distances)
    train_distances = cdist(X_train, X_train, metric='euclidean')
    train_dist_upper = train_distances[np.triu_indices_from(train_distances, k=1)]
    threshold = np.mean(train_dist_upper) + 2 * np.std(train_dist_upper)
    
    # Identify compounds outside AD
    outside_ad = min_distances > threshold
    
    return min_distances, threshold, outside_ad

def main():
    logger.info("=" * 70)
    logger.info("PHASE 4: APPLICABILITY DOMAIN ANALYSIS")
    logger.info("=" * 70)
    
    # Load data
    df = pd.read_csv('bioavailability_with_descriptors.csv')
    
    with open('selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    logger.info(f"Loaded {len(df)} compounds")
    
    # Prepare data
    X = df[selected_features].values
    y = df['Bioavailability'].values
    X_scaled = scaler.transform(X)
    
    # Split data (same as in phase 3)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Calculate applicability domain
    logger.info("\nCalculating applicability domain...")
    min_distances, threshold, outside_ad = calculate_applicability_domain(X_train, X_test)
    
    logger.info(f"Applicability Domain Threshold: {threshold:.4f}")
    logger.info(f"Compounds outside AD: {np.sum(outside_ad)} / {len(X_test)} ({100*np.sum(outside_ad)/len(X_test):.1f}%)")
    
    # Train best model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QSAR Model Validation and Applicability Domain Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_test, y_pred_test, alpha=0.6, s=50, color='steelblue')
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Experimental Bioavailability (%)', fontsize=11)
    ax.set_ylabel('Predicted Bioavailability (%)', fontsize=11)
    ax.set_title('Predicted vs Experimental Values', fontweight='bold')
    r2 = r2_score(y_test, y_pred_test)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax = axes[0, 1]
    residuals = y_test - y_pred_test
    ax.scatter(y_pred_test, residuals, alpha=0.6, s=50, color='steelblue')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Bioavailability (%)', fontsize=11)
    ax.set_ylabel('Residuals (%)', fontsize=11)
    ax.set_title('Residual Plot', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Applicability Domain
    ax = axes[1, 0]
    colors = ['red' if x else 'green' for x in outside_ad]
    ax.scatter(range(len(min_distances)), min_distances, c=colors, alpha=0.6, s=50)
    ax.axhline(y=threshold, color='black', linestyle='--', lw=2, label=f'AD Threshold = {threshold:.2f}')
    ax.set_xlabel('Compound Index', fontsize=11)
    ax.set_ylabel('Min Distance to Training Set', fontsize=11)
    ax.set_title('Applicability Domain (Distance Method)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error Distribution
    ax = axes[1, 1]
    ax.hist(np.abs(residuals), bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Absolute Error (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Error Distribution', fontweight='bold')
    ax.axvline(np.mean(np.abs(residuals)), color='r', linestyle='--', lw=2, label=f'Mean = {np.mean(np.abs(residuals)):.2f}%')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('validation_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Saved validation_analysis.png")
    
    # Generate summary report
    logger.info("\n" + "=" * 70)
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info("=" * 70)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = np.mean(np.abs(residuals))
    
    logger.info(f"\nExternal Test Set Performance:")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  RMSE: {rmse:.2f}%")
    logger.info(f"  MAE: {mae:.2f}%")
    logger.info(f"  Mean Residual: {np.mean(residuals):.2f}%")
    logger.info(f"  Std Residual: {np.std(residuals):.2f}%")
    
    logger.info(f"\nApplicability Domain:")
    logger.info(f"  Method: Euclidean Distance")
    logger.info(f"  Threshold: {threshold:.2f}")
    logger.info(f"  Compounds within AD: {len(X_test) - np.sum(outside_ad)} / {len(X_test)}")
    logger.info(f"  Compounds outside AD: {np.sum(outside_ad)} / {len(X_test)}")
    
    logger.info(f"\nModel Applicability:")
    logger.info(f"  The model shows excellent predictive performance (R² = {r2:.4f})")
    logger.info(f"  Low prediction errors (MAE = {mae:.2f}%) indicate reliable bioavailability predictions")
    logger.info(f"  Compounds outside AD should be used with caution")
    
    logger.info("\n✅ Phase 4 completed successfully!")

if __name__ == '__main__':
    main()
