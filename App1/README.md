# Fetal Health Monitoring System

A production-ready Streamlit application with integrated machine learning models for fetal health classification using Cardiotocography (CTG) data.

## Features

### 🤖 Multiple ML Models
- **Random Forest** - Ensemble method with high accuracy
- **Gradient Boosting** - Sequential ensemble for excellent performance
- **Logistic Regression** - Fast and interpretable baseline
- **K-Nearest Neighbors** - Instance-based learning

### 🎯 Key Capabilities
- Real-time CTG analysis
- Model selection and comparison
- Automated training pipeline
- Performance monitoring
- Prediction history tracking
- Interactive visualizations
- Clinical recommendations

### 📊 Performance Targets
- Model Accuracy: ≥ 90%
- Inference Time: < 50ms
- Prediction Time: < 2 seconds
- Page Load: < 3 seconds

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the repository**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your data**
   - Ensure you have the fetal health dataset at `Data/fetal_health.csv`
   - The CSV should contain the following columns:
     - baseline value
     - accelerations
     - fetal_movement
     - uterine_contractions
     - prolongued_decelerations
     - abnormal_short_term_variability
     - mean_value_of_short_term_variability
     - percentage_of_time_with_abnormal_long_term_variability
     - mean_value_of_long_term_variability
     - histogram_width
     - histogram_min
     - histogram_max
     - histogram_number_of_peaks
     - histogram_number_of_zeroes
     - histogram_mode
     - histogram_mean
     - histogram_median
     - histogram_variance
     - histogram_tendency
     - fetal_health (target: 1=Normal, 2=Suspect, 3=Pathological)

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### First-Time Setup

1. **Initialize Models**
   - Click "🚀 Initialize Models" in the sidebar
   - The system will automatically:
     - Load existing trained models (if available)
     - Train new models from the data (if no models found)
     - Save models for future use

2. **Select a Model**
   - Choose your preferred ML model from the dropdown
   - Options: Random Forest, Gradient Boosting, Logistic Regression, KNN

### Making Predictions

1. Navigate to "🔬 Prediction" page
2. Enter patient information:
   - Patient ID
   - Gestational Age
   - Maternal Age
   - Examination Date

3. Input CTG parameters across four categories:
   - 💓 Heart Rate
   - 📈 Accelerations & Decelerations
   - 🔄 Variability
   - 📊 Histogram Features

4. Click "🔍 Analyze Fetal Health"
5. Review results:
   - Prediction classification
   - Confidence level
   - Probability distribution
   - Feature importance
   - Clinical recommendations

### Retraining Models

If you want to retrain models with updated data:
1. Update your `Data/fetal_health.csv` file
2. Click "🔄 Retrain Models" in the sidebar
3. Wait for training to complete

## Model Training Details

### Training Pipeline

The application implements the complete training pipeline from the notebook:

1. **Data Loading & Preprocessing**
   - Loads data from CSV
   - Splits into train/test sets (80/20)
   - Stratified sampling to maintain class distribution
   - Feature scaling using StandardScaler

2. **Hyperparameter Tuning**
   - **Logistic Regression**: GridSearchCV with 5-fold CV
   - **KNN**: GridSearchCV with 5-fold CV
   - **Random Forest**: RandomizedSearchCV (50 iterations)
   - **Gradient Boosting**: RandomizedSearchCV (50 iterations)

3. **Model Evaluation**
   - Accuracy metrics
   - Confusion matrices
   - Classification reports
   - Cross-validation scores

4. **Model Persistence**
   - Models saved to `models/` directory
   - Includes trained models, scaler, and metadata
   - Automatic loading on subsequent runs

### Model Parameters

Each model is tuned with the following parameter ranges:

**Logistic Regression**
- C: [0.001, 0.01, 0.1, 1, 10, 100]
- Penalty: ['l2']
- Solver: ['lbfgs', 'liblinear']

**K-Nearest Neighbors**
- n_neighbors: [3, 5, 7, 9, 11, 15]
- weights: ['uniform', 'distance']
- metric: ['euclidean', 'manhattan', 'minkowski']

**Random Forest**
- n_estimators: 100-500
- max_depth: [10, 20, 30, None]
- min_samples_split: 2-20
- min_samples_leaf: 1-10

**Gradient Boosting**
- n_estimators: 100-500
- learning_rate: 0.01-0.31
- max_depth: 3-10
- min_samples_split: 2-20

## Directory Structure

```
.
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Data/
│   └── fetal_health.csv       # Training dataset
└── models/                     # Trained models (auto-generated)
    ├── logistic_regression.pkl
    ├── knn.pkl
    ├── random_forest.pkl
    ├── gradient_boosting.pkl
    ├── scaler.pkl
    └── metadata.json
```

## Features by Page

### 🏠 Home
- System overview
- Model performance comparison
- Getting started guide
- Model accuracy visualization

### 🔬 Prediction
- Patient information input
- CTG parameter entry (organized in tabs)
- Real-time prediction
- Probability distribution charts
- Feature importance visualization
- Clinical recommendations

### 📜 History
- Prediction history table
- Summary statistics
- Distribution charts
- CSV export functionality
- Clear history option

### ⚡ Performance
- Response time tracking
- Performance metrics
- Threshold monitoring
- Detailed operation logs

### 📖 Help
- System documentation
- Model descriptions
- Performance benchmarks
- CTG parameter guide
- Important disclaimers

## Important Notes

### ⚠️ Disclaimer
This system is for **demonstration and research purposes only**.

- NOT FDA approved
- NOT for clinical decision making
- Always consult qualified healthcare professionals
- Results should be verified by trained specialists

### Data Privacy
- All data is processed locally
- No patient data is transmitted externally
- Session data is cleared when browser is closed

### Performance Optimization
- Models are cached after first load
- Predictions run in < 50ms
- Automatic performance monitoring
- Threshold alerts for slow operations

## Troubleshooting

### Models won't train
- Check that `Data/fetal_health.csv` exists and has correct format
- Ensure all required columns are present
- Verify sufficient disk space for model files

### Slow predictions
- Check system resources (CPU, RAM)
- Consider using simpler models (Logistic Regression, KNN)
- Review performance dashboard for bottlenecks

### Missing visualizations
- Ensure plotly is installed: `pip install plotly`
- Clear browser cache
- Check console for JavaScript errors

## Technical Specifications

### Frontend
- Framework: Streamlit
- Charts: Plotly
- Styling: Custom CSS (Dark Theme)

### Backend
- ML Framework: scikit-learn
- Data Processing: pandas, numpy
- Model Storage: pickle

### Models
- All models use stratified cross-validation
- Automatic hyperparameter tuning
- Feature scaling via StandardScaler
- Class balancing via stratified sampling

## Support

For issues or questions:
1. Check this README
2. Review the Help page in the application
3. Examine training logs in console output

## License

For demonstration and educational purposes.

## Acknowledgments

Based on the Fetal Health Classification dataset and implements best practices from the structured notebook approach.
