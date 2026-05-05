"""
Image preprocessing utilities for DICOM images.

Handles normalization, enhancement, filtering, and resizing of medical images.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from src.config import config


class ImagePreprocessor:
    """Main preprocessing class for medical images."""
    
    def __init__(self, config_obj=None):
        """Initialize preprocessor with configuration.
        
        Args:
            config_obj: Configuration object (uses global config if None)
        """
        self.config = config_obj or config.preprocessing
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Complete preprocessing pipeline.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image
        """
        # Resize
        image = self.resize(image)
        
        # Enhance contrast
        image = self.apply_clahe(image)
        
        # Denoise with bilateral filter
        image = self.apply_bilateral_filter(image)
        
        # Normalize
        image = self.normalize(image)
        
        return image
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        target_h = self.config.target_height
        target_w = self.config.target_width
        
        if (height, width) != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        return image
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            image: Input image (8-bit or 16-bit)
            
        Returns:
            CLAHE-enhanced image
        """
        # Convert to 8-bit if needed
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            img_min, img_max = image.min(), image.max()
            image = ((image - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size)
        )
        
        return clahe.apply(image)
    
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for edge-preserving denoising.
        
        Args:
            image: Input image
            
        Returns:
            Filtered image
        """
        # Ensure 8-bit for bilateral filter
        if image.dtype != np.uint8:
            img_min, img_max = image.min(), image.max()
            image = ((image - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
        
        return cv2.bilateralFilter(
            image,
            d=self.config.bilateral_diameter,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space
        )
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for smoothing.
        
        Args:
            image: Input image
            
        Returns:
            Blurred image
        """
        sigma = self.config.gaussian_sigma
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using specified method.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        method = self.config.normalization_method
        
        if method == "zscore":
            mean = np.mean(image)
            std = np.std(image)
            return (image - mean) / (std + 1e-8)
        
        elif method == "minmax":
            img_min = np.min(image)
            img_max = np.max(image)
            return (image - img_min) / (img_max - img_min + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def apply_morphological_operations(image: np.ndarray,
                                       operation: str = "close",
                                       kernel_size: int = 5) -> np.ndarray:
        """Apply morphological operations (open, close, erode, dilate).
        
        Args:
            image: Input binary or grayscale image
            operation: Type of operation ('open', 'close', 'erode', 'dilate')
            kernel_size: Size of the morphological kernel
            
        Returns:
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        operations = {
            "open": cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel),
            "close": cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel),
            "erode": cv2.erode(image, kernel),
            "dilate": cv2.dilate(image, kernel),
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        return operations[operation]
    
    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization.
        
        Args:
            image: Input image
            
        Returns:
            Equalized image
        """
        if image.dtype != np.uint8:
            img_min, img_max = image.min(), image.max()
            image = ((image - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
        
        return cv2.equalizeHist(image)
    
    @staticmethod
    def apply_gaussian_blur_static(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur (static method).
        
        Args:
            image: Input image
            sigma: Gaussian sigma
            
        Returns:
            Blurred image
        """
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
