# Aneurysm Detection

A hybrid deep learning pipeline for detecting cerebral aneurysms from CTA (CT Angiography) images using CNN feature extraction combined with gradient boosting classification.


## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/NeerajPratapSingh/Aneurysm-Detection.git
cd Aneurysm-Detection

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Run Web Interface
```bash
streamlit run src/app/ui.py
```
Access at `http://localhost:8501`

#### 2. Run Demo Notebook
```bash
jupyter notebook notebooks/demo.ipynb
```

#### 3. Use as Python Module
```python
from src.pipeline.hybrid_pipeline import HybridPipeline
from src.utils.dicom_loader import load_dicom

# Load DICOM
dicom_array = load_dicom('path/to/dicom.dcm')

# Create and train pipeline
pipeline = HybridPipeline()
pipeline.train(X_train, y_train)

# Predict
prediction = pipeline.predict(dicom_array)
```

## 📊 Key Features

### Preprocessing
- **DICOM Loading**: Full support for DICOM files with Hounsfield unit conversion
- **Image Enhancement**: CLAHE, bilateral filtering, Gaussian blur
- **Slice Selection**: Multiple strategies (center, max_variance, manual)
- **Normalization**: Z-score and min-max normalization

### Models
- **CNN Extractors**: ResNet50, VGG16, Inception
- **Classifiers**: Gradient Boosting, XGBoost, Logistic Regression, KNN, Random Forest
- **Ensemble**: Support for multiple model combinations

### Pipeline
- **Hybrid Pipeline**: CNN features + Gradient Boosting
- **Original Pipeline**: Hand-crafted features + Classical ML
- **Cross-validation**: K-fold and stratified split support

## 📈 Model Architecture

### Hybrid Approach
```
DICOM Image → Preprocessing → CNN Feature Extraction → Boosting Classifier → Prediction
```

### Original Approach
```
DICOM Image → Preprocessing → Hand-crafted Features → Classical ML → Prediction
```

## 🔧 Configuration

Edit `src/config.py` to customize:
- Model hyperparameters
- Preprocessing settings
- Training parameters
- Data paths

## 📝 Metadata Format

The `data/metadata.csv` should contain:
```csv
patient_id,age,gender,dicom_path,aneurysm_present
PT001,45,M,data/sample/PT001.dcm,1
PT002,52,F,data/sample/PT002.dcm,0
```

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

## 📚 Documentation

Detailed documentation for each module:
- [Preprocessing Guide](docs/preprocessing.md)
- [Model Guide](docs/models.md)
- [Pipeline Guide](docs/pipeline.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## 👨‍💻 Authors

- **Neeraj Pratap Singh** - Initial implementation

## 🙏 Acknowledgments

- PyTorch and TensorFlow communities
- DICOM imaging standards
- Medical imaging research community

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Last Updated**: 2026-05-05
