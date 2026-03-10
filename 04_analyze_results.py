#!/usr/bin/env python3
"""
QSAR Model for Drug Bioavailability Prediction
Phase 4: Analyze and visualize model results

This script creates visualizations for model performance,
feature importance, and prediction accuracy.

Author: Pharmaceutical R&D Portfolio
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class ResultAnalyzer:
    """Analyze and visualize QSAR model results"""
    
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.predictions = {}
    
    def load_data(self):
        """Load data and models"""
        logger.info("Loading data and models...")
        
        # Load data
        self.df = pd.read_csv('bioavailability_with_descriptors.csv')
        
        # Prepare features and target
        exclude_cols = ['name', 'smiles', 'inchi', 'source', 'oral_bioavailability']
        descriptor_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        self.X = self.df[descriptor_cols].values
        self.y = self.df['oral_bioavailability'].values
        
        # Split data (same as training)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Load scaler
        with open('scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Scale test data
        self.X_test = self.scaler.transform(self.X_test)
        
        # Load models
        model_names = ['linear_regression', 'random_forest', 'gradient_boosting', 'svr']
        for name in model_names:
            with open(f'model_{name}.pkl', 'rb') as f:
                self.models[name] = pickle.load(f)
                # Get predictions
                self.predictions[name] = self.models[name].predict(self.X_test)
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def plot_model_comparison(self):
        """Create model comparison visualization"""
        logger.info("Creating model comparison plot...")
        
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        results = []
        for name, y_pred in self.predictions.items():
            results.append({
                'Model': name.replace('_', ' ').title(),
                'R² Score': r2_score(self.y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                'MAE': mean_absolute_error(self.y_test, y_pred)
            })
        
        results_df = pd.DataFrame(results)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # R² Score
        axes[0].bar(results_df['Model'], results_df['R² Score'], color='steelblue', alpha=0.7)
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('Model R² Score Comparison', fontsize=13, fontweight='bold')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].tick_params(axis='x', rotation=45)
        
        # RMSE
        axes[1].bar(results_df['Model'], results_df['RMSE'], color='coral', alpha=0.7)
        axes[1].set_ylabel('RMSE (%)', fontsize=12)
        axes[1].set_title('Model RMSE Comparison', fontsize=13, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        # MAE
        axes[2].bar(results_df['Model'], results_df['MAE'], color='mediumseagreen', alpha=0.7)
        axes[2].set_ylabel('MAE (%)', fontsize=12)
        axes[2].set_title('Model MAE Comparison', fontsize=13, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("Saved model_comparison.png")
        plt.close()
    
    def plot_predictions_vs_actual(self):
        """Create predictions vs actual values plots"""
        logger.info("Creating predictions vs actual plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        model_names = list(self.predictions.keys())
        colors = ['steelblue', 'coral', 'mediumseagreen', 'gold']
        
        for idx, (name, y_pred) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(self.y_test, y_pred, alpha=0.6, s=100, color=colors[idx])
            
            # Perfect prediction line
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Labels and title
            ax.set_xlabel('Actual Bioavailability (%)', fontsize=11)
            ax.set_ylabel('Predicted Bioavailability (%)', fontsize=11)
            ax.set_title(f'{name.replace("_", " ").title()} Predictions', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        logger.info("Saved predictions_vs_actual.png")
        plt.close()
    
    def plot_residuals(self):
        """Create residual plots"""
        logger.info("Creating residual plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        colors = ['steelblue', 'coral', 'mediumseagreen', 'gold']
        
        for idx, (name, y_pred) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            # Calculate residuals
            residuals = self.y_test - y_pred
            
            # Residual plot
            ax.scatter(y_pred, residuals, alpha=0.6, s=100, color=colors[idx])
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            
            # Labels and title
            ax.set_xlabel('Predicted Bioavailability (%)', fontsize=11)
            ax.set_ylabel('Residuals (%)', fontsize=11)
            ax.set_title(f'{name.replace("_", " ").title()} Residuals', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
        logger.info("Saved residuals_plot.png")
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        logger.info("Creating feature importance plots...")
        
        # Get feature names
        exclude_cols = ['name', 'smiles', 'inchi', 'source', 'oral_bioavailability']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Random Forest
        rf_importance = self.models['random_forest'].feature_importances_
        top_indices_rf = np.argsort(rf_importance)[-10:]
        
        axes[0].barh(np.array(feature_cols)[top_indices_rf], rf_importance[top_indices_rf], color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Importance', fontsize=11)
        axes[0].set_title('Random Forest - Top 10 Features', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Gradient Boosting
        gb_importance = self.models['gradient_boosting'].feature_importances_
        top_indices_gb = np.argsort(gb_importance)[-10:]
        
        axes[1].barh(np.array(feature_cols)[top_indices_gb], gb_importance[top_indices_gb], color='coral', alpha=0.7)
        axes[1].set_xlabel('Importance', fontsize=11)
        axes[1].set_title('Gradient Boosting - Top 10 Features', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        logger.info("Saved feature_importance.png")
        plt.close()
    
    def plot_bioavailability_distribution(self):
        """Plot bioavailability distribution"""
        logger.info("Creating bioavailability distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.y, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Oral Bioavailability (%)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Distribution of Oral Bioavailability', fontsize=12, fontweight='bold')
        axes[0].axvline(self.y.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {self.y.mean():.1f}%')
        axes[0].axvline(np.median(self.y), color='green', linestyle='--', lw=2, label=f'Median: {np.median(self.y):.1f}%')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Box plot
        categories = []
        values = []
        for val in self.y:
            if val < 30:
                categories.append('Low (<30%)')
            elif val < 80:
                categories.append('Medium (30-80%)')
            else:
                categories.append('High (>80%)')
            values.append(val)
        
        box_data = [self.y[np.array(categories) == cat] for cat in ['Low (<30%)', 'Medium (30-80%)', 'High (>80%)']]
        bp = axes[1].boxplot(box_data, labels=['Low (<30%)', 'Medium (30-80%)', 'High (>80%)'], patch_artist=True)
        
        for patch, color in zip(bp['boxes'], ['lightcoral', 'lightyellow', 'lightgreen']):
            patch.set_facecolor(color)
        
        axes[1].set_ylabel('Oral Bioavailability (%)', fontsize=11)
        axes[1].set_title('Bioavailability Categories', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('bioavailability_distribution.png', dpi=300, bbox_inches='tight')
        logger.info("Saved bioavailability_distribution.png")
        plt.close()

def main():
    """Main execution function"""
    logger.info("Starting result analysis...")
    
    analyzer = ResultAnalyzer()
    analyzer.load_data()
    
    # Create visualizations
    analyzer.plot_model_comparison()
    analyzer.plot_predictions_vs_actual()
    analyzer.plot_residuals()
    analyzer.plot_feature_importance()
    analyzer.plot_bioavailability_distribution()
    
    logger.info("Analysis complete!")
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print("\nGenerated plots:")
    print("  ✓ model_comparison.png - Model performance comparison")
    print("  ✓ predictions_vs_actual.png - Predictions vs actual values")
    print("  ✓ residuals_plot.png - Model residuals analysis")
    print("  ✓ feature_importance.png - Top features for tree models")
    print("  ✓ bioavailability_distribution.png - Data distribution")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
