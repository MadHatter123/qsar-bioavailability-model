#!/usr/bin/env python3
"""
QSAR Model for Drug Bioavailability Prediction
Phase 3: Train and evaluate machine learning models

This script trains multiple ML models to predict oral bioavailability:
- Linear Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression
- Neural Network

Author: Pharmaceutical R&D Portfolio
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import pickle
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QSARModelTrainer:
    """Train and evaluate QSAR models for bioavailability prediction"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_and_prepare_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data and prepare features and target
        
        Args:
            filepath: Path to CSV file with descriptors
            
        Returns:
            Tuple of (X, y) arrays
        """
        logger.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Select descriptor columns (exclude metadata)
        exclude_cols = ['name', 'smiles', 'inchi', 'source', 'oral_bioavailability']
        descriptor_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[descriptor_cols].values
        y = df['oral_bioavailability'].values
        
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Features: {len(descriptor_cols)}")
        logger.info(f"Target range: {y.min():.1f}% - {y.max():.1f}%")
        
        return X, y, descriptor_cols
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """
        Split data into train and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data for testing
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"Train set: {self.X_train.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
    
    def train_linear_regression(self):
        """Train linear regression model"""
        logger.info("Training Linear Regression...")
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        self.models['Linear Regression'] = model
        self.results['Linear Regression'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'y_pred': y_pred_test
        }
        
        logger.info(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"  Test RMSE: {test_rmse:.2f}%, MAE: {test_mae:.2f}%")
    
    def train_random_forest(self):
        """Train random forest model"""
        logger.info("Training Random Forest...")
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        self.models['Random Forest'] = model
        self.results['Random Forest'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'y_pred': y_pred_test,
            'feature_importance': model.feature_importances_
        }
        
        logger.info(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"  Test RMSE: {test_rmse:.2f}%, MAE: {test_mae:.2f}%")
    
    def train_gradient_boosting(self):
        """Train gradient boosting model"""
        logger.info("Training Gradient Boosting...")
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        self.models['Gradient Boosting'] = model
        self.results['Gradient Boosting'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'y_pred': y_pred_test,
            'feature_importance': model.feature_importances_
        }
        
        logger.info(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"  Test RMSE: {test_rmse:.2f}%, MAE: {test_mae:.2f}%")
    
    def train_svr(self):
        """Train Support Vector Regression model"""
        logger.info("Training Support Vector Regression...")
        
        model = SVR(kernel='rbf', C=100, epsilon=0.1)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        self.models['SVR'] = model
        self.results['SVR'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'y_pred': y_pred_test
        }
        
        logger.info(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"  Test RMSE: {test_rmse:.2f}%, MAE: {test_mae:.2f}%")
    
    def train_all_models(self):
        """Train all models"""
        logger.info("="*60)
        logger.info("TRAINING ML MODELS")
        logger.info("="*60)
        
        self.train_linear_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_svr()
        
        logger.info("="*60)
    
    def save_models(self):
        """Save trained models to disk"""
        logger.info("Saving models...")
        
        for name, model in self.models.items():
            filename = f"model_{name.lower().replace(' ', '_')}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"  Saved {filename}")
        
        # Save scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info("  Saved scaler.pkl")
    
    def generate_report(self):
        """Generate performance report"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train R²': [v['train_r2'] for v in self.results.values()],
            'Test R²': [v['test_r2'] for v in self.results.values()],
            'RMSE (%)': [v['test_rmse'] for v in self.results.values()],
            'MAE (%)': [v['test_mae'] for v in self.results.values()]
        })
        
        print("\n" + results_df.to_string(index=False))
        
        # Find best model
        best_model = results_df.loc[results_df['Test R²'].idxmax()]
        print(f"\n✓ Best Model: {best_model['Model']}")
        print(f"  Test R²: {best_model['Test R²']:.4f}")
        print(f"  RMSE: {best_model['RMSE (%)']:.2f}%")
        print(f"  MAE: {best_model['MAE (%)']:.2f}%")
        
        print("\n" + "="*60)
        
        # Save results to CSV
        results_df.to_csv('model_performance.csv', index=False)
        logger.info("Results saved to model_performance.csv")

def main():
    """Main execution function"""
    logger.info("Starting QSAR model training...")
    
    # Initialize trainer
    trainer = QSARModelTrainer()
    
    # Load and prepare data
    X, y, feature_cols = trainer.load_and_prepare_data('bioavailability_with_descriptors.csv')
    
    # Split data
    trainer.split_data(X, y)
    
    # Train all models
    trainer.train_all_models()
    
    # Save models
    trainer.save_models()
    
    # Generate report
    trainer.generate_report()

if __name__ == "__main__":
    main()
