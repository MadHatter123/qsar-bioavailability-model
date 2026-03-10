#!/usr/bin/env python3
"""
QSAR Model for Drug Bioavailability Prediction
Phase 2: Calculate molecular descriptors

This script calculates molecular descriptors for each compound
using RDKit, including Lipinski's Rule of Five descriptors
and other important physicochemical properties.

Author: Pharmaceutical R&D Portfolio
Date: 2026
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import logging
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MolecularDescriptorCalculator:
    """Calculate molecular descriptors from SMILES strings"""
    
    def calculate_descriptors(self, smiles: str) -> Dict:
        """
        Calculate molecular descriptors for a compound
        
        Args:
            smiles: SMILES string representation of the molecule
            
        Returns:
            Dictionary with descriptor names and values
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            descriptors = {}
            
            # Lipinski's Rule of Five descriptors
            descriptors['MW'] = Descriptors.MolWt(mol)
            descriptors['LogP'] = Crippen.MolLogP(mol)
            descriptors['HBD'] = Descriptors.NumHDonors(mol)
            descriptors['HBA'] = Descriptors.NumHAcceptors(mol)
            descriptors['RotBonds'] = Descriptors.NumRotatableBonds(mol)
            
            # Topological Polar Surface Area
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            
            # Molar refractivity
            descriptors['MolRefr'] = Crippen.MolMR(mol)
            
            # Ring descriptors
            descriptors['NumRings'] = Descriptors.RingCount(mol)
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
            descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            
            # Atom count descriptors
            descriptors['NumAtoms'] = Descriptors.HeavyAtomCount(mol)
            descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
            
            # Molecular complexity
            descriptors['BertzCT'] = Descriptors.BertzCT(mol)
            
            # Charge descriptors
            descriptors['FormalCharge'] = Chem.GetFormalCharge(mol)
            
            # Exact molecular weight
            descriptors['ExactMolWt'] = Descriptors.ExactMolWt(mol)
            
            # Fraction sp3 carbons
            try:
                descriptors['FractionCsp3'] = Descriptors.FractionCsp3(mol)
            except:
                descriptors['FractionCsp3'] = 0
            
            # Additional PEOE descriptors
            try:
                descriptors['PEOE_VSA1'] = Descriptors.PEOE_VSA1(mol)
                descriptors['PEOE_VSA2'] = Descriptors.PEOE_VSA2(mol)
                descriptors['PEOE_VSA3'] = Descriptors.PEOE_VSA3(mol)
            except:
                descriptors['PEOE_VSA1'] = 0
                descriptors['PEOE_VSA2'] = 0
                descriptors['PEOE_VSA3'] = 0
            
            # Kappa descriptors
            try:
                descriptors['Kappa1'] = Descriptors.Kappa1(mol)
                descriptors['Kappa2'] = Descriptors.Kappa2(mol)
            except:
                descriptors['Kappa1'] = 0
                descriptors['Kappa2'] = 0
            
            # Chi descriptors
            try:
                descriptors['Chi0'] = Descriptors.Chi0(mol)
                descriptors['Chi1'] = Descriptors.Chi1(mol)
            except:
                descriptors['Chi0'] = 0
                descriptors['Chi1'] = 0
            
            return descriptors
            
        except Exception as e:
            logger.error(f"Error calculating descriptors for SMILES {smiles}: {str(e)}")
            return None
    
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire dataset and calculate descriptors
        
        Args:
            df: DataFrame with compound data
            
        Returns:
            DataFrame with calculated descriptors
        """
        logger.info(f"Processing {len(df)} compounds...")
        
        descriptor_dicts = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            smiles = row['smiles']
            descriptors = self.calculate_descriptors(smiles)
            
            if descriptors is not None:
                descriptor_dicts.append(descriptors)
                valid_indices.append(idx)
            else:
                logger.warning(f"Could not process compound {row['name']} (index {idx})")
        
        # Create DataFrame from descriptors
        descriptors_df = pd.DataFrame(descriptor_dicts)
        
        # Keep only rows with valid descriptors
        df_filtered = df.iloc[valid_indices].reset_index(drop=True)
        descriptors_df = descriptors_df.reset_index(drop=True)
        
        # Combine original data with descriptors
        result_df = pd.concat([df_filtered, descriptors_df], axis=1)
        
        logger.info(f"Successfully processed {len(result_df)} compounds")
        logger.info(f"Calculated {len(descriptors_df.columns)} descriptors")
        
        return result_df

def main():
    """Main execution function"""
    logger.info("Starting molecular descriptor calculation...")
    
    # Load bioavailability data
    df = pd.read_csv('bioavailability_data.csv')
    logger.info(f"Loaded {len(df)} compounds from bioavailability_data.csv")
    
    # Calculate descriptors
    calculator = MolecularDescriptorCalculator()
    df_with_descriptors = calculator.process_dataset(df)
    
    # Save to CSV
    output_file = 'bioavailability_with_descriptors.csv'
    df_with_descriptors.to_csv(output_file, index=False)
    logger.info(f"Data with descriptors saved to {output_file}")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("DESCRIPTOR CALCULATION SUMMARY")
    print("="*60)
    print(f"\nTotal compounds processed: {len(df_with_descriptors)}")
    print(f"Total descriptors calculated: {len(df_with_descriptors.columns) - 5}")  # Minus original 5 columns
    
    print(f"\nDescriptor Statistics (sample):")
    descriptor_cols = [col for col in df_with_descriptors.columns if col not in ['name', 'smiles', 'inchi', 'source', 'oral_bioavailability']]
    
    for col in descriptor_cols[:10]:  # Show first 10 descriptors
        print(f"\n{col}:")
        print(f"  Mean: {df_with_descriptors[col].mean():.2f}")
        print(f"  Std Dev: {df_with_descriptors[col].std():.2f}")
        print(f"  Min: {df_with_descriptors[col].min():.2f}")
        print(f"  Max: {df_with_descriptors[col].max():.2f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
