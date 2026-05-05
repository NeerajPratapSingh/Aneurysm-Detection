"""
Metadata handling utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class MetadataHandler:
    """Handle patient and DICOM metadata."""
    
    @staticmethod
    def load_metadata(filepath: str) -> pd.DataFrame:
        """Load metadata from CSV file.
        
        Args:
            filepath: Path to metadata CSV
            
        Returns:
            DataFrame with metadata
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        
        return pd.read_csv(filepath)
    
    @staticmethod
    def save_metadata(df: pd.DataFrame, filepath: str) -> None:
        """Save metadata to CSV file.
        
        Args:
            df: DataFrame with metadata
            filepath: Path to save CSV
        """
        df.to_csv(filepath, index=False)
    
    @staticmethod
    def get_statistics(metadata_df: pd.DataFrame) -> dict:
        """Get statistics from metadata.
        
        Args:
            metadata_df: Metadata DataFrame
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_samples": len(metadata_df),
            "positive_samples": len(metadata_df[metadata_df['aneurysm_present'] == 1]),
            "negative_samples": len(metadata_df[metadata_df['aneurysm_present'] == 0]),
        }
        
        stats["positive_ratio"] = stats["positive_samples"] / stats["total_samples"]
        
        if 'age' in metadata_df.columns:
            stats["age_mean"] = metadata_df['age'].mean()
            stats["age_std"] = metadata_df['age'].std()
        
        return stats
    
    @staticmethod
    def split_train_test(metadata_df: pd.DataFrame,
                        test_size: float = 0.2,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split metadata into train and test sets (stratified).
        
        Args:
            metadata_df: Metadata DataFrame
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        if 'aneurysm_present' not in metadata_df.columns:
            raise ValueError("'aneurysm_present' column required for stratified split")
        
        train_df, test_df = train_test_split(
            metadata_df,
            test_size=test_size,
            random_state=random_state,
            stratify=metadata_df['aneurysm_present']
        )
        
        return train_df, test_df
    
    @staticmethod
    def split_train_val_test(metadata_df: pd.DataFrame,
                            val_size: float = 0.1,
                            test_size: float = 0.2,
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split metadata into train, validation, and test sets.
        
        Args:
            metadata_df: Metadata DataFrame
            val_size: Proportion for validation set
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        temp_size = val_size + test_size
        train_df, temp_df = train_test_split(
            metadata_df,
            test_size=temp_size,
            random_state=random_state,
            stratify=metadata_df['aneurysm_present']
        )
        
        # Second split: val vs test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / temp_size,
            random_state=random_state,
            stratify=temp_df['aneurysm_present']
        )
        
        return train_df, val_df, test_df
    
    @staticmethod
    def create_metadata_from_directory(directory: str,
                                      labels: Optional[Dict[str, int]] = None) -> pd.DataFrame:
        """Create metadata DataFrame from directory of DICOM files.
        
        Args:
            directory: Path to directory containing DICOM files
            labels: Dictionary mapping patient IDs to labels (optional)
            
        Returns:
            Metadata DataFrame
        """
        dicom_files = list(Path(directory).glob('*.dcm'))
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {directory}")
        
        data = []
        for dcm_file in dicom_files:
            patient_id = dcm_file.stem
            label = labels.get(patient_id, -1) if labels else -1
            
            data.append({
                'patient_id': patient_id,
                'dicom_path': str(dcm_file),
                'aneurysm_present': label,
            })
        
        return pd.DataFrame(data)
