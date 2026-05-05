"""
Intelligent DICOM slice selection strategies.

Provides multiple methods to select the most relevant slice from a 3D DICOM volume.
"""

import numpy as np
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod


class SliceSelector(ABC):
    """Abstract base class for slice selection strategies."""
    
    @abstractmethod
    def select(self, volume: np.ndarray) -> Tuple[np.ndarray, int]:
        """Select slice(s) from volume.
        
        Args:
            volume: 3D array of shape (depth, height, width)
            
        Returns:
            Tuple of (selected_slice, slice_index)
        """
        pass


class CenterSliceSelector(SliceSelector):
    """Selects the center slice of the volume."""
    
    def select(self, volume: np.ndarray) -> Tuple[np.ndarray, int]:
        """Select center slice.
        
        Args:
            volume: 3D array of shape (depth, height, width)
            
        Returns:
            Center slice and its index
        """
        center_idx = volume.shape[0] // 2
        return volume[center_idx], center_idx


class MaxVarianceSliceSelector(SliceSelector):
    """Selects slice with maximum variance (most information)."""
    
    def select(self, volume: np.ndarray) -> Tuple[np.ndarray, int]:
        """Select slice with maximum variance.
        
        Args:
            volume: 3D array of shape (depth, height, width)
            
        Returns:
            Slice with max variance and its index
        """
        # Calculate variance for each slice
        variances = np.array([np.var(volume[i]) for i in range(volume.shape[0])])
        
        # Find slice with maximum variance
        max_var_idx = np.argmax(variances)
        return volume[max_var_idx], max_var_idx


class MultiSliceSelector(SliceSelector):
    """Selects multiple equally-spaced slices from the volume."""
    
    def __init__(self, num_slices: int = 3):
        """Initialize multi-slice selector.
        
        Args:
            num_slices: Number of slices to select
        """
        self.num_slices = num_slices
    
    def select(self, volume: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Select multiple slices.
        
        Args:
            volume: 3D array of shape (depth, height, width)
            
        Returns:
            Stacked slices and list of indices
        """
        depth = volume.shape[0]
        indices = np.linspace(0, depth - 1, self.num_slices, dtype=int)
        
        selected_slices = np.array([volume[i] for i in indices])
        return selected_slices, indices.tolist()


class ManualSliceSelector(SliceSelector):
    """Allows manual selection of specific slice(s)."""
    
    def __init__(self, slice_indices: Optional[List[int]] = None):
        """Initialize manual selector.
        
        Args:
            slice_indices: Indices of slices to select
        """
        self.slice_indices = slice_indices or []
    
    def select(self, volume: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Select manually specified slices.
        
        Args:
            volume: 3D array of shape (depth, height, width)
            
        Returns:
            Selected slices and their indices
        """
        if not self.slice_indices:
            # Fallback to center if no indices specified
            center_idx = volume.shape[0] // 2
            self.slice_indices = [center_idx]
        
        # Ensure indices are within bounds
        valid_indices = [i for i in self.slice_indices if 0 <= i < volume.shape[0]]
        
        if not valid_indices:
            raise ValueError(f"No valid slice indices. Volume depth: {volume.shape[0]}")
        
        selected_slices = np.array([volume[i] for i in valid_indices])
        return selected_slices, valid_indices


class AdaptiveSliceSelector(SliceSelector):
    """Adaptively selects slices based on image content.
    
    Selects center slice plus adjacent slices if they contain relevant features.
    """
    
    def __init__(self, num_adjacent: int = 1, variance_threshold: float = 0.1):
        """Initialize adaptive selector.
        
        Args:
            num_adjacent: Number of adjacent slices to consider
            variance_threshold: Minimum relative variance to include adjacent slice
        """
        self.num_adjacent = num_adjacent
        self.variance_threshold = variance_threshold
    
    def select(self, volume: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Select center slice and relevant adjacent slices.
        
        Args:
            volume: 3D array of shape (depth, height, width)
            
        Returns:
            Selected slices and their indices
        """
        # Start with center slice
        center_idx = volume.shape[0] // 2
        center_var = np.var(volume[center_idx])
        
        selected_indices = [center_idx]
        
        # Check adjacent slices
        depth = volume.shape[0]
        for offset in range(1, self.num_adjacent + 1):
            # Check above
            if center_idx + offset < depth:
                above_var = np.var(volume[center_idx + offset])
                if above_var > center_var * self.variance_threshold:
                    selected_indices.append(center_idx + offset)
            
            # Check below
            if center_idx - offset >= 0:
                below_var = np.var(volume[center_idx - offset])
                if below_var > center_var * self.variance_threshold:
                    selected_indices.insert(0, center_idx - offset)
        
        selected_slices = np.array([volume[i] for i in sorted(set(selected_indices))])
        return selected_slices, sorted(set(selected_indices))


def get_slice_selector(strategy: str, **kwargs) -> SliceSelector:
    """Factory function to get appropriate slice selector.
    
    Args:
        strategy: Selection strategy name
        **kwargs: Additional arguments for specific selectors
        
    Returns:
        SliceSelector instance
        
    Raises:
        ValueError: If strategy is unknown
    """
    strategies = {
        "center": CenterSliceSelector,
        "max_variance": MaxVarianceSliceSelector,
        "multi": MultiSliceSelector,
        "manual": ManualSliceSelector,
        "adaptive": AdaptiveSliceSelector,
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
    
    selector_class = strategies[strategy]
    return selector_class(**kwargs)
