"""
Streamlit web interface for Aneurysm Detection System.
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.dicom_loader import DICOMLoader
from src.preprocessing.preprocessing import ImagePreprocessor
from src.preprocessing.visualization import Visualizer


def main():
    """Main Streamlit app."""
    
    st.set_page_config(page_title="Aneurysm Detection", layout="wide")
    
    # Header
    st.title("🧠 Aneurysm Detection System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select Page",
            ["Home", "Upload & Predict", "Preprocessing Demo", "About"]
        )
    
    # Pages
    if page == "Home":
        show_home()
    elif page == "Upload & Predict":
        show_prediction()
    elif page == "Preprocessing Demo":
        show_preprocessing_demo()
    elif page == "About":
        show_about()


def show_home():
    """Show home page."""
    st.header("Welcome to Aneurysm Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### About This System
        
        This is an advanced medical imaging analysis system designed to detect
        cerebral aneurysms from CTA (CT Angiography) images.
        
        **Features:**
        - CNN-based feature extraction
        - Boosting-based classification
        - Real-time predictions
        - Comprehensive preprocessing
        
        """)
    
    with col2:
        st.markdown("""
        ### How to Use
        
        1. **Upload & Predict**: Upload DICOM files for prediction
        2. **Preprocessing Demo**: See preprocessing effects
        3. **About**: Learn more about the system
        
        **Note**: This system requires trained models. Please ensure models are
        available in the outputs directory.
        """)


def show_prediction():
    """Show prediction page."""
    st.header("Upload & Predict")
    
    st.markdown("""
    Upload a DICOM file to get a prediction. The system will:
    1. Load and preprocess the image
    2. Extract CNN features
    3. Use trained classifier for prediction
    """)
    
    uploaded_file = st.file_uploader("Choose a DICOM file", type=["dcm"])
    
    if uploaded_file is not None:
        try:
            # Save temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load DICOM
            st.info("Loading DICOM file...")
            dicom_array = DICOMLoader.load_dicom(temp_path)
            
            # Display info
            st.success(f"✓ DICOM loaded. Shape: {dicom_array.shape}")
            
            # Show slice
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original DICOM Slice")
                st.image(dicom_array / dicom_array.max(), use_column_width=True, clamp=True)
            
            with col2:
                st.subheader("Statistics")
                st.metric("Min Value", f"{dicom_array.min():.2f}")
                st.metric("Max Value", f"{dicom_array.max():.2f}")
                st.metric("Mean Value", f"{dicom_array.mean():.2f}")
                st.metric("Std Dev", f"{dicom_array.std():.2f}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def show_preprocessing_demo():
    """Show preprocessing demonstration."""
    st.header("Preprocessing Demo")
    
    st.markdown("""
    This demo shows the effect of various preprocessing techniques.
    """)
    
    # Create sample image
    sample_image = np.random.randn(512, 512) * 30 + 100
    sample_image = np.clip(sample_image, 0, 255).astype(np.uint8)
    
    preprocessor = ImagePreprocessor()
    
    # Display preprocessing steps
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(sample_image, use_column_width=True)
    
    with col2:
        st.subheader("After CLAHE")
        clahe_image = preprocessor.apply_clahe(sample_image)
        st.image(clahe_image, use_column_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("After Bilateral Filter")
        filtered = preprocessor.apply_bilateral_filter(sample_image)
        st.image(filtered, use_column_width=True)
    
    with col4:
        st.subheader("Preprocessed")
        preprocessed = preprocessor.preprocess(sample_image)
        # Normalize for display
        preprocessed_display = (preprocessed - preprocessed.min()) / (preprocessed.max() - preprocessed.min())
        st.image(preprocessed_display, use_column_width=True)


def show_about():
    """Show about page."""
    st.header("About")
    
    st.markdown("""


if __name__ == "__main__":
    main()
