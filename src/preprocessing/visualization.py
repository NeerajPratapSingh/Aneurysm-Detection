"""
Visualization utilities for DICOM images and model results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional


class Visualizer:
    """Visualization utilities for medical images."""
    
    @staticmethod
    def plot_image(image: np.ndarray, title: str = "Image",
                   cmap: str = "gray", figsize: Tuple[int, int] = (6, 6)):
        """Plot a single image.
        
        Args:
            image: Image array
            title: Plot title
            cmap: Colormap
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_comparison(images: List[np.ndarray], titles: List[str],
                       cmap: str = "gray", figsize: Tuple[int, int] = (12, 4)):
        """Plot multiple images side by side.
        
        Args:
            images: List of image arrays
            titles: List of titles
            cmap: Colormap
            figsize: Figure size
        """
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=figsize)
        
        if num_images == 1:
            axes = [axes]
        
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_histogram(image: np.ndarray, title: str = "Histogram",
                      bins: int = 256, figsize: Tuple[int, int] = (8, 4)):
        """Plot image histogram.
        
        Args:
            image: Image array
            title: Plot title
            bins: Number of histogram bins
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        if len(image.shape) == 2:
            # Grayscale
            hist = np.histogram(image.flatten(), bins=bins)
            plt.plot(hist[1][:-1], hist[0])
        else:
            # Color
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = np.histogram(image[:, :, i].flatten(), bins=bins)
                plt.plot(hist[1][:-1], hist[0], color=color, label=f"Channel {i}")
            plt.legend()
        
        plt.title(title)
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_with_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int],
                               title: str = "Image with BBox",
                               figsize: Tuple[int, int] = (8, 8)):
        """Plot image with bounding box.
        
        Args:
            image: Image array
            bbox: Bounding box (x, y, width, height)
            title: Plot title
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap="gray")
        
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_slices(volume: np.ndarray, indices: List[int] = None,
                   figsize: Tuple[int, int] = (12, 4)):
        """Plot multiple slices from a 3D volume.
        
        Args:
            volume: 3D volume array
            indices: Slice indices to plot (if None, plots every 10th slice)
            figsize: Figure size
        """
        if indices is None:
            indices = list(range(0, volume.shape[0], max(1, volume.shape[0] // 5)))
        
        num_slices = len(indices)
        fig, axes = plt.subplots(1, num_slices, figsize=figsize)
        
        if num_slices == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, indices):
            ax.imshow(volume[idx], cmap="gray")
            ax.set_title(f"Slice {idx}")
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_prediction_results(image: np.ndarray, prediction: dict,
                               figsize: Tuple[int, int] = (10, 6)):
        """Plot image with prediction results.
        
        Args:
            image: Input image
            prediction: Prediction dictionary with keys: 'aneurysm', 'confidence'
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap="gray")
        
        # Add prediction text
        title = f"Aneurysm: {prediction['aneurysm']} | Confidence: {prediction['confidence']:.2%}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis("off")
        
        # Color based on prediction
        color = 'red' if prediction['aneurysm'] else 'green'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)
        
        plt.tight_layout()
        plt.show()
