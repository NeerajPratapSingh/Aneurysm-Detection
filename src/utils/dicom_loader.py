"""
DICOM file loading and processing utilities.
"""

import numpy as np
from typing import Tuple, Optional
import pydicom
from pathlib import Path


class DICOMLoader:
    """Load and process DICOM files."""
    
    # Hounsfield Unit ranges for different tissue types
    HU_RANGES = {
        "brain": (-40, 80),
        "bone": (400, 3071),
        "lung": (-500, -200),
        "soft_tissue": (40, 80),
    }
    
    @staticmethod
    def load_dicom(filepath: str) -> np.ndarray:
        """Load DICOM file and convert to numpy array.
        
        Args:
            filepath: Path to DICOM file
            
        Returns:
            DICOM data as numpy array
        """
        dcm = pydicom.dcmread(filepath)
        
        # Get pixel array
        pixel_array = dcm.pixel_array.astype(np.float32)
        
        # Convert to Hounsfield Units if slope and intercept available
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        
        return pixel_array
    
    @staticmethod
    def load_dicom_series(folder: str) -> np.ndarray:
        """Load a series of DICOM files from folder.
        
        Args:
            folder: Path to folder containing DICOM files
            
        Returns:
            3D array of stacked DICOM slices (depth, height, width)
        """
        folder_path = Path(folder)
        dicom_files = sorted(folder_path.glob('*.dcm'))
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {folder}")
        
        slices = []
        for dcm_file in dicom_files:
            dcm_array = DICOMLoader.load_dicom(str(dcm_file))
            slices.append(dcm_array)
        
        return np.stack(slices, axis=0)
    
    @staticmethod
    def apply_window_level(pixel_array: np.ndarray, window: int = 400,
                          level: int = 40) -> np.ndarray:
        """Apply window/level adjustment to DICOM data.
        
        Args:
            pixel_array: Input pixel array
            window: Window width
            level: Window level
            
        Returns:
            Windowed image
        """
        lower_bound = level - (window // 2)
        upper_bound = level + (window // 2)
        
        windowed = np.clip(pixel_array, lower_bound, upper_bound)
        windowed = (windowed - lower_bound) / (upper_bound - lower_bound) * 255
        
        return windowed.astype(np.uint8)
    
    @staticmethod
    def normalize_hu(pixel_array: np.ndarray, tissue_type: str = "brain") -> np.ndarray:
        """Normalize Hounsfield Units to standard range.
        
        Args:
            pixel_array: Input pixel array in HU
            tissue_type: Type of tissue for appropriate windowing
            
        Returns:
            Normalized array in [0, 1] range
        """
        if tissue_type not in DICOMLoader.HU_RANGES:
            raise ValueError(f"Unknown tissue type: {tissue_type}")
        
        hu_min, hu_max = DICOMLoader.HU_RANGES[tissue_type]
        normalized = np.clip(pixel_array, hu_min, hu_max)
        normalized = (normalized - hu_min) / (hu_max - hu_min)
        
        return normalized
    
    @staticmethod
    def get_dicom_metadata(filepath: str) -> dict:
        """Extract metadata from DICOM file.
        
        Args:
            filepath: Path to DICOM file
            
        Returns:
            Dictionary of metadata
        """
        dcm = pydicom.dcmread(filepath)
        
        metadata = {
            "patient_id": getattr(dcm, 'PatientID', None),
            "patient_age": getattr(dcm, 'PatientAge', None),
            "patient_sex": getattr(dcm, 'PatientSex', None),
            "modality": getattr(dcm, 'Modality', None),
            "series_description": getattr(dcm, 'SeriesDescription', None),
            "image_position": getattr(dcm, 'ImagePositionPatient', None),
            "image_orientation": getattr(dcm, 'ImageOrientationPatient', None),
            "pixel_spacing": getattr(dcm, 'PixelSpacing', None),
            "slice_thickness": getattr(dcm, 'SliceThickness', None),
            "rescale_slope": getattr(dcm, 'RescaleSlope', None),
            "rescale_intercept": getattr(dcm, 'RescaleIntercept', None),
        }
        
        return metadata
