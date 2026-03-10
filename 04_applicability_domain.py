#!/usr/bin/env python3
"""
Phase 4: Applicability Domain (AD) Analysis

Method: Euclidean distance in descriptor space
- Calculate mean distance in training set
- Set AD threshold = mean + 2*std
- Evaluate test compounds against threshold
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_ad_distance(X_train, X_test, scaler):
    """Calculate Euclidean distance from test compounds to training set center"""
    # Scale data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate center of training set
    train_center = np.mean(X_train_scaled, axis=0)
    
    # Calculate distances from test compounds to center
    distances = np.linalg.norm(X_test_scaled - train_center, axis=1)
    
    # Calculate AD threshold (mean + 2*std of training distances)
    train_distances = np.linalg.norm(X_train_scaled - train_center, axis=1)
    ad_threshold = np.mean(train_distances) + 2 * np.std(train_distances)
    
    return distances, ad_threshold, train_distances

def main():
    logger.info("="*70)
    logger.info("Phase 4: Applicability Domain Analysis")
    logger.info("="*70)
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv('bioavailability_selected_features.csv')
    
    # Load scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Separate features and target
    X = df.iloc[:, 3:].values
    y = df['Bioavailability_%'].values
    
    # Split into train (80%) and test (20%)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)), test_size=0.2, random_state=42
    )
    
    # Calculate AD
    logger.info("Calculating applicability domain...")
    test_distances, ad_threshold, train_distances = calculate_ad_distance(X_train, X_test, scaler)
    
    # Analyze results
    within_ad = np.sum(test_distances <= ad_threshold)
    outside_ad = np.sum(test_distances > ad_threshold)
    percent_within = (within_ad / len(test_distances)) * 100
    
    logger.info(f"\nApplicability Domain Results:")
    logger.info(f"  AD Threshold: {ad_threshold:.2f}")
    logger.info(f"  Training set distance (mean ± std): {np.mean(train_distances):.2f} ± {np.std(train_distances):.2f}")
    logger.info(f"  Test set distance (mean ± std): {np.mean(test_distances):.2f} ± {np.std(test_distances):.2f}")
    logger.info(f"  Compounds within AD: {within_ad}/{len(test_distances)} ({percent_within:.1f}%)")
    logger.info(f"  Compounds outside AD: {outside_ad}/{len(test_distances)} ({100-percent_within:.1f}%)")
    
    # Create AD report
    ad_report = pd.DataFrame({
        'Compound_Index': idx_test,
        'Distance': test_distances,
        'Within_AD': test_distances <= ad_threshold,
        'Bioavailability_%': y_test,
    })
    
    ad_report.to_csv('applicability_domain_report.csv', index=False)
    logger.info("\n✓ AD report saved to applicability_domain_report.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("APPLICABILITY DOMAIN SUMMARY")
    print("="*70)
    print(f"\nAD Threshold: {ad_threshold:.2f}")
    print(f"Compounds within AD: {within_ad}/{len(test_distances)} ({percent_within:.1f}%)")
    print(f"\nAD Statistics:")
    print(f"  Training set distances:")
    print(f"    Mean: {np.mean(train_distances):.2f}")
    print(f"    Std: {np.std(train_distances):.2f}")
    print(f"    Min: {np.min(train_distances):.2f}")
    print(f"    Max: {np.max(train_distances):.2f}")
    print(f"\n  Test set distances:")
    print(f"    Mean: {np.mean(test_distances):.2f}")
    print(f"    Std: {np.std(test_distances):.2f}")
    print(f"    Min: {np.min(test_distances):.2f}")
    print(f"    Max: {np.max(test_distances):.2f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
