# QSAR Model for Drug Bioavailability Prediction

## Overview

This project implements a **Quantitative Structure-Activity Relationship (QSAR)** model to predict oral bioavailability of pharmaceutical compounds. The model uses machine learning algorithms trained on molecular descriptors calculated from chemical structures.

**Key Features:**
- 85 FDA-approved drugs with known bioavailability values
- 24 molecular descriptors (Lipinski's Rule of Five + topological descriptors)
- 4 ML algorithms (Linear Regression, Random Forest, Gradient Boosting, SVR)
- Comprehensive analysis and visualization of model performance

## Project Structure

```
qsar-bioavailability-model/
├── 01_collect_bioavailability_data.py    # Data collection and preparation
├── 02_calculate_descriptors.py           # Molecular descriptor calculation
├── 03_train_ml_models.py                 # Model training and evaluation
├── 04_analyze_results.py                 # Results analysis and visualization
├── bioavailability_data.csv              # Raw bioavailability dataset
├── bioavailability_with_descriptors.csv  # Dataset with calculated descriptors
├── model_performance.csv                 # Model performance metrics
├── model_*.pkl                           # Trained model files
├── scaler.pkl                            # Feature scaler
├── *.png                                 # Visualization plots
└── README.md                             # This file
```

## Workflow

### Phase 1: Data Collection (`01_collect_bioavailability_data.py`)

Collects bioavailability data from literature and public databases:
- **89 compounds** with oral bioavailability values (5-99%)
- **SMILES representations** for chemical structure encoding
- **Bioavailability categories**: Low (<30%), Medium (30-80%), High (>80%)

**Output:** `bioavailability_data.csv`

### Phase 2: Descriptor Calculation (`02_calculate_descriptors.py`)

Calculates 24 molecular descriptors using RDKit:

**Lipinski's Rule of Five:**
- Molecular Weight (MW)
- Lipophilicity (LogP)
- Hydrogen Bond Donors (HBD)
- Hydrogen Bond Acceptors (HBA)
- Rotatable Bonds

**Additional Descriptors:**
- Topological Polar Surface Area (TPSA)
- Molar Refractivity
- Ring descriptors (aromatic, saturated, aliphatic)
- Atom count descriptors
- Molecular complexity (BertzCT)
- Charge and electronic descriptors

**Output:** `bioavailability_with_descriptors.csv`

### Phase 3: Model Training (`03_train_ml_models.py`)

Trains 4 machine learning models:

1. **Linear Regression**
   - Simple baseline model
   - Test R²: -0.2223, MAE: 26.39%

2. **Random Forest**
   - Ensemble method with 100 trees
   - Test R²: -0.5622, MAE: 29.44%

3. **Gradient Boosting**
   - Sequential tree building
   - Test R²: -0.9190, MAE: 31.14%

4. **Support Vector Regression (SVR)**
   - Non-linear kernel (RBF)
   - Test R²: -0.6314, MAE: 30.38%

**Output:** Trained models (`model_*.pkl`), scaler (`scaler.pkl`), metrics (`model_performance.csv`)

### Phase 4: Results Analysis (`04_analyze_results.py`)

Generates comprehensive visualizations:

- **model_comparison.png** - R², RMSE, and MAE comparison across models
- **predictions_vs_actual.png** - Predicted vs actual bioavailability scatter plots
- **residuals_plot.png** - Residual analysis for each model
- **feature_importance.png** - Top 10 important features for tree-based models
- **bioavailability_distribution.png** - Data distribution and category breakdown

## Model Performance

| Model | Train R² | Test R² | RMSE (%) | MAE (%) |
|-------|----------|---------|----------|---------|
| Linear Regression | 0.4488 | -0.2223 | 31.00 | 26.39 |
| Random Forest | 0.8036 | -0.5622 | 35.05 | 29.44 |
| Gradient Boosting | 0.9026 | -0.9190 | 38.84 | 31.14 |
| SVR | 0.6473 | -0.6314 | 35.82 | 30.38 |

**Best Model:** Linear Regression (lowest test error)

## Key Findings

### Feature Importance (Random Forest)
Top predictive features for bioavailability:
1. Topological Polar Surface Area (TPSA)
2. Molecular Weight (MW)
3. Hydrogen Bond Donors (HBD)
4. LogP (Lipophilicity)
5. Number of Rotatable Bonds

### Bioavailability Distribution
- **High (>80%):** 26 compounds (30.6%)
- **Medium (30-80%):** 49 compounds (57.6%)
- **Low (<30%):** 14 compounds (16.5%)
- **Mean:** 61.0% | **Median:** 60.0%

## Installation

### Requirements
```bash
pip install pandas numpy scikit-learn rdkit matplotlib seaborn
```

### Python Version
- Python 3.8+

## Usage

### Run Complete Pipeline
```bash
# Phase 1: Collect data
python3 01_collect_bioavailability_data.py

# Phase 2: Calculate descriptors
python3 02_calculate_descriptors.py

# Phase 3: Train models
python3 03_train_ml_models.py

# Phase 4: Analyze results
python3 04_analyze_results.py
```

### Use Trained Model for Prediction
```python
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

# Load model and scaler
with open('model_linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example: Predict bioavailability for a compound
smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
mol = Chem.MolFromSmiles(smiles)

# Calculate descriptors
descriptors = [
    Descriptors.MolWt(mol),
    Crippen.MolLogP(mol),
    Descriptors.NumHDonors(mol),
    Descriptors.NumHAcceptors(mol),
    Descriptors.NumRotatableBonds(mol),
    # ... add all 24 descriptors
]

# Scale and predict
X = scaler.transform([descriptors])
bioavailability = model.predict(X)[0]
print(f"Predicted Bioavailability: {bioavailability:.1f}%")
```

## Scientific Background

### QSAR Methodology
QSAR models establish quantitative relationships between molecular structure and biological activity. The approach:
1. Encodes chemical structure as numerical descriptors
2. Trains regression models on known activity data
3. Predicts activity for new compounds

### Lipinski's Rule of Five
Predicts drug-likeness based on:
- MW ≤ 500 Da
- LogP ≤ 5
- HBD ≤ 5
- HBA ≤ 10

All compounds in this dataset satisfy Lipinski's criteria.

### Bioavailability Factors
Oral bioavailability depends on:
- **Absorption:** TPSA, LogP, HBD/HBA
- **Distribution:** Lipophilicity, protein binding
- **Metabolism:** Structural features (CYP450 substrates)
- **Excretion:** Molecular weight, polarity

## Limitations & Future Work

### Current Limitations
1. **Small dataset (85 compounds)** - leads to overfitting
2. **Limited descriptor set** - could include 3D descriptors
3. **No metabolic stability data** - important for bioavailability
4. **Single species (human)** - human data only

### Recommendations for Improvement
1. **Expand dataset** - collect 500+ compounds with diverse structures
2. **Add 3D descriptors** - conformer-dependent properties
3. **Include ADME data** - metabolism, protein binding, clearance
4. **Implement ensemble methods** - combine multiple models
5. **Cross-validation** - use k-fold CV for robust evaluation
6. **Feature selection** - identify most important descriptors
7. **Applicability domain** - define model validity range

## References

1. **Lipinski, C. A., et al.** (2001). "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings." *Advanced Drug Delivery Reviews*, 46(1-3), 3-26.

2. **Delaney, J. S.** (2004). "ESOL: Estimating aqueous solubility directly from molecular structure." *Journal of Chemical Information and Computer Sciences*, 44(3), 1000-1005.

3. **Ertl, P., et al.** (2000). "Fast calculation of molecular polar surface area as a sum of fragment-based contributions and its application to the prediction of drug transport properties." *Journal of Medicinal Chemistry*, 43(20), 3714-3717.

4. **Veber, D. F., et al.** (2002). "Molecular properties that influence the oral bioavailability of drug candidates." *Journal of Medicinal Chemistry*, 45(12), 2615-2623.

## Author

**Pharmaceutical R&D Portfolio**
- Created: 2026
- Purpose: Demonstrate QSAR modeling for drug discovery
- Target: Pharmaceutical R&D positions

## License

MIT License - Open for educational and research purposes

## Contact & Support

For questions or improvements, please refer to the project documentation or contact the author.

---

**Note:** This model is for educational and research purposes. For drug development applications, consult with pharmaceutical experts and validate predictions experimentally.
