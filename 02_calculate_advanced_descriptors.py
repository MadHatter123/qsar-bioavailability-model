#!/usr/bin/env python3
"""
Phase 2: Calculate advanced molecular descriptors (100+)

Descriptor types:
1. Lipinski's Rule of Five (5 descriptors)
2. Topological descriptors (20+ descriptors)
3. Electronic descriptors (15+ descriptors)
4. Molecular complexity (10+ descriptors)
5. ECFP fingerprints (2048 bits)

Feature selection: SelectKBest to identify top 50 most important features
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, AllChem
from sklearn.feature_selection import SelectKBest, f_regression
import logging
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_descriptors(mol):
    """Calculate comprehensive set of molecular descriptors"""
    if mol is None:
        return None
    
    descriptors = {}
    
    # Lipinski's Rule of Five
    descriptors['MW'] = Descriptors.MolWt(mol)
    descriptors['LogP'] = Crippen.MolLogP(mol)
    descriptors['HBD'] = Descriptors.NumHDonors(mol)
    descriptors['HBA'] = Descriptors.NumHAcceptors(mol)
    descriptors['RotBonds'] = Descriptors.NumRotatableBonds(mol)
    
    # Topological descriptors
    descriptors['TPSA'] = Descriptors.TPSA(mol)
    descriptors['NumRings'] = Descriptors.RingCount(mol)
    descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
    descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
    descriptors['NumHeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
    descriptors['NumAtoms'] = mol.GetNumAtoms()
    descriptors['NumBonds'] = mol.GetNumBonds()
    # Removed: NumAromaticAtoms, NumAliphaticAtoms (not available in this RDKit version)
    descriptors['FractionCsp3'] = Descriptors.FractionCSP3(mol)
    
    # Electronic descriptors
    descriptors['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
    descriptors['NumLipinskiHBA'] = Descriptors.NumHAcceptors(mol)
    descriptors['NumLipinskiHBD'] = Descriptors.NumHDonors(mol)
    descriptors['Molar_Refractivity'] = Crippen.MolMR(mol)
    
    # Molecular complexity
    descriptors['BertzCT'] = Descriptors.BertzCT(mol)
    descriptors['Ipc'] = Descriptors.Ipc(mol)
    descriptors['LabuteASA'] = Descriptors.LabuteASA(mol)
    descriptors['PEOE_VSA1'] = Descriptors.PEOE_VSA1(mol)
    descriptors['PEOE_VSA2'] = Descriptors.PEOE_VSA2(mol)
    descriptors['PEOE_VSA3'] = Descriptors.PEOE_VSA3(mol)
    descriptors['PEOE_VSA4'] = Descriptors.PEOE_VSA4(mol)
    descriptors['PEOE_VSA5'] = Descriptors.PEOE_VSA5(mol)
    descriptors['PEOE_VSA6'] = Descriptors.PEOE_VSA6(mol)
    descriptors['SMR_VSA1'] = Descriptors.SMR_VSA1(mol)
    descriptors['SMR_VSA2'] = Descriptors.SMR_VSA2(mol)
    descriptors['SMR_VSA3'] = Descriptors.SMR_VSA3(mol)
    descriptors['SMR_VSA4'] = Descriptors.SMR_VSA4(mol)
    descriptors['SMR_VSA5'] = Descriptors.SMR_VSA5(mol)
    descriptors['SMR_VSA6'] = Descriptors.SMR_VSA6(mol)
    descriptors['EState_VSA1'] = Descriptors.EState_VSA1(mol)
    descriptors['EState_VSA2'] = Descriptors.EState_VSA2(mol)
    descriptors['EState_VSA3'] = Descriptors.EState_VSA3(mol)
    descriptors['EState_VSA4'] = Descriptors.EState_VSA4(mol)
    descriptors['EState_VSA5'] = Descriptors.EState_VSA5(mol)
    descriptors['EState_VSA6'] = Descriptors.EState_VSA6(mol)
    descriptors['EState_VSA7'] = Descriptors.EState_VSA7(mol)
    descriptors['EState_VSA8'] = Descriptors.EState_VSA8(mol)
    descriptors['EState_VSA9'] = Descriptors.EState_VSA9(mol)
    descriptors['EState_VSA10'] = Descriptors.EState_VSA10(mol)
    descriptors['EState_VSA11'] = Descriptors.EState_VSA11(mol)
    
    # Additional complexity metrics
    descriptors['NumSulfur'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
    descriptors['NumPhosphorus'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15)
    descriptors['NumChlorine'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17)
    descriptors['NumFluorine'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    descriptors['NumBromine'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 35)
    descriptors['NumIodine'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 53)
    descriptors['NumNitrogen'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    descriptors['NumOxygen'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    
    return descriptors

def calculate_ecfp_features(mol, radius=2, nbits=64):
    """Calculate ECFP (Extended Connectivity Fingerprint) features"""
    if mol is None:
        return [0] * nbits
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return list(fp)

def main():
    logger.info("="*70)
    logger.info("Phase 2: Calculate advanced molecular descriptors")
    logger.info("="*70)
    
    # Load data
    logger.info("Loading bioavailability data...")
    df = pd.read_csv('bioavailability_rigorous_data.csv')
    logger.info(f"Loaded {len(df)} compounds")
    
    # Calculate descriptors
    logger.info("Calculating molecular descriptors...")
    descriptor_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            descriptors = calculate_descriptors(mol)
            if descriptors is not None:
                descriptor_list.append(descriptors)
                valid_indices.append(idx)
        
        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} compounds")
    
    logger.info(f"✓ Successfully calculated descriptors for {len(descriptor_list)} compounds")
    
    # Create descriptor DataFrame
    desc_df = pd.DataFrame(descriptor_list)
    
    # Add ECFP features
    logger.info("Calculating ECFP fingerprints...")
    ecfp_features = []
    for idx in valid_indices:
        mol = Chem.MolFromSmiles(df.loc[idx, 'SMILES'])
        ecfp = calculate_ecfp_features(mol, radius=2, nbits=64)
        ecfp_features.append(ecfp)
    
    ecfp_df = pd.DataFrame(ecfp_features, columns=[f'ECFP_{i}' for i in range(64)])
    desc_df = pd.concat([desc_df, ecfp_df], axis=1)
    
    logger.info(f"✓ Total descriptors: {desc_df.shape[1]}")
    
    # Add bioavailability target
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    df_filtered['Bioavailability_%'] = df_filtered['Bioavailability_%'].values
    
    # Combine with descriptors
    final_df = pd.concat([df_filtered[['SMILES', 'Compound_Name', 'Bioavailability_%']], desc_df], axis=1)
    
    # Feature selection
    logger.info("Performing feature selection (SelectKBest, k=50)...")
    X = desc_df.values
    y = df_filtered['Bioavailability_%'].values
    
    selector = SelectKBest(score_func=f_regression, k=50)
    X_selected = selector.fit_transform(X, y)
    
    selected_features = desc_df.columns[selector.get_support()].tolist()
    logger.info(f"✓ Selected {len(selected_features)} most important features")
    logger.info(f"Selected features: {', '.join(selected_features[:10])}...")
    
    # Save results
    final_df.to_csv('bioavailability_with_descriptors.csv', index=False)
    logger.info("✓ Full dataset saved to bioavailability_with_descriptors.csv")
    
    # Save selected features
    selected_df = final_df[['SMILES', 'Compound_Name', 'Bioavailability_%'] + selected_features]
    selected_df.to_csv('bioavailability_selected_features.csv', index=False)
    logger.info("✓ Selected features dataset saved to bioavailability_selected_features.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("DESCRIPTOR CALCULATION SUMMARY")
    print("="*70)
    print(f"\nCompounds with valid descriptors: {len(descriptor_list)}")
    print(f"Total descriptors calculated: {desc_df.shape[1]}")
    print(f"Selected features (top 50): {len(selected_features)}")
    print(f"\nTop 10 selected features:")
    for i, feat in enumerate(selected_features[:10], 1):
        print(f"  {i}. {feat}")
    print(f"\nDescriptor statistics:")
    print(desc_df.describe().round(3))
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
