"""
Fetal Health Monitoring System v2.0
Streamlit Cloud Optimized - Fast Ensemble ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import warnings
import pickle
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Performance monitoring
def track_performance(func):
    """Decorator to track function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
        
        st.session_state.performance_metrics.append({
            'function': func.__name__,
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now()
        })
        
        return result
    return wrapper

# Model Manager Class
class FetalHealthModelManager:
    """Fast Ensemble Model Manager for Streamlit Cloud"""
    
    def __init__(self, data_path="Data/fetal_health.csv", models_dir="models"):
        self.data_path = data_path
        self.models_dir = Path(models_dir)
        
        # Create models directory if it doesn't exist
        try:
            self.models_dir.mkdir(exist_ok=True)
            print(f"✓ Models directory ready: {self.models_dir}")
        except Exception as e:
            print(f"Note: {e}")
        
        self.models = {}
        self.scaler = None
        self.model_metadata = {}
        self.feature_names = []
        
        self.available_models = {
            'Ensemble Voting': 'voting_ensemble',
            'Random Forest': 'random_forest',
            'Gradient Boosting': 'gradient_boosting'
        }
    
    def check_data_exists(self):
        """Check if data file exists"""
        if os.path.exists(self.data_path):
            print(f"✓ Data file found: {self.data_path}")
            return True
        else:
            print(f"✗ Data file not found: {self.data_path}")
            return False
    
    def load_and_preprocess_data(self):
        """Load and preprocess the fetal health dataset"""
        if not self.check_data_exists():
            raise FileNotFoundError(f"Data file not found at: {self.data_path}\nPlease ensure Data/fetal_health.csv exists in your repository.")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Separate features and target
        X = df.drop('fetal_health', axis=1)
        y = df['fetal_health']
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
    
    def train_random_forest_fast(self, X_train, y_train):
        """Train optimized Random Forest"""
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)
        return rf
    
    def train_gradient_boosting_fast(self, X_train, y_train):
        """Train optimized Gradient Boosting"""
        print("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train, y_train)
        return gb
    
    def train_voting_ensemble(self, X_train, y_train):
        """Train Voting Ensemble"""
        print("Training Voting Ensemble...")
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        voting = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft',
            n_jobs=-1
        )
        voting.fit(X_train, y_train)
        return voting
    
    def models_exist(self):
        """Check if all required model files exist"""
        required_files = [
            'random_forest.pkl',
            'gradient_boosting.pkl', 
            'voting_ensemble.pkl',
            'scaler.pkl'
        ]
        
        all_exist = all((self.models_dir / f).exists() for f in required_files)
        
        if all_exist:
            print("✓ All model files found")
        else:
            print("✗ Some model files missing")
            for f in required_files:
                path = self.models_dir / f
                status = "✓" if path.exists() else "✗"
                print(f"  {status} {f}")
        
        return all_exist
    
    def train_all_models(self):
        """Train all models"""
        print("\n" + "="*50)
        print("Starting Model Training Pipeline")
        print("="*50)
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, scaler, feature_names = self.load_and_preprocess_data()
        
        self.feature_names = feature_names
        self.scaler = scaler
        
        results = {}
        
        # Train Random Forest
        rf_model = self.train_random_forest_fast(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
        self.models['random_forest'] = rf_model
        results['Random Forest'] = {'accuracy': rf_acc}
        print(f"✓ Random Forest: {rf_acc:.4f}")
        
        # Train Gradient Boosting
        gb_model = self.train_gradient_boosting_fast(X_train, y_train)
        gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
        self.models['gradient_boosting'] = gb_model
        results['Gradient Boosting'] = {'accuracy': gb_acc}
        print(f"✓ Gradient Boosting: {gb_acc:.4f}")
        
        # Train Voting Ensemble
        voting_model = self.train_voting_ensemble(X_train, y_train)
        voting_acc = accuracy_score(y_test, voting_model.predict(X_test))
        self.models['voting_ensemble'] = voting_model
        results['Ensemble Voting'] = {'accuracy': voting_acc}
        print(f"✓ Voting Ensemble: {voting_acc:.4f}")
        
        # Save models
        self.save_all_models()
        
        # Save metadata
        self.model_metadata = {
            'training_date': datetime.now().isoformat(),
            'results': results,
            'feature_names': feature_names
        }
        self._save_metadata()
        
        print("="*50)
        print("Training Complete!")
        print("="*50 + "\n")
        
        return results
    
    def save_all_models(self):
        """Save all models"""
        try:
            for model_key, model in self.models.items():
                filepath = self.models_dir / f"{model_key}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✓ Saved {model_key}")
            
            # Save scaler
            scaler_path = self.models_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✓ Saved scaler")
            
        except Exception as e:
            print(f"✗ Error saving models: {str(e)}")
            raise
    
    def _save_metadata(self):
        """Save metadata"""
        try:
            metadata_path = self.models_dir / "metadata.json"
            metadata = {
                'training_date': self.model_metadata['training_date'],
                'feature_names': self.model_metadata['feature_names'],
                'results': {
                    model: {'accuracy': float(metrics['accuracy'])}
                    for model, metrics in self.model_metadata['results'].items()
                }
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Saved metadata")
        except Exception as e:
            print(f"Note: Could not save metadata: {e}")
    
    def load_all_models(self):
        """Load all models"""
        try:
            print("\nLoading models...")
            
            # Load each model
            for model_key in ['random_forest', 'gradient_boosting', 'voting_ensemble']:
                filepath = self.models_dir / f"{model_key}.pkl"
                with open(filepath, 'rb') as f:
                    self.models[model_key] = pickle.load(f)
                print(f"✓ Loaded {model_key}")
            
            # Load scaler
            scaler_path = self.models_dir / "scaler.pkl"
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Loaded scaler")
            
            # Load metadata
            metadata_path = self.models_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                    self.feature_names = self.model_metadata.get('feature_names', [])
                print(f"✓ Loaded metadata")
            
            print("✓ All models loaded successfully!\n")
            return True
            
        except Exception as e:
            print(f"✗ Error loading models: {str(e)}")
            return False
    
    def predict(self, features, model_name='voting_ensemble'):
        """Make prediction"""
        start_time = time.time()
        
        # Prepare features
        feature_array = np.array([features[col] for col in self.feature_names]).reshape(1, -1)
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_array)
        
        # Get model
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict(feature_scaled)[0]
        probabilities = model.predict_proba(feature_scaled)[0]
        
        # Map prediction
        class_names = {1.0: 'Normal', 2.0: 'Suspect', 3.0: 'Pathological'}
        prediction_label = class_names.get(prediction, 'Unknown')
        
        # Get feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            ))
        elif model_name == 'voting_ensemble':
            rf_imp = self.models['random_forest'].feature_importances_
            gb_imp = self.models['gradient_boosting'].feature_importances_
            avg_imp = (rf_imp + gb_imp) / 2
            feature_importance = dict(sorted(
                zip(self.feature_names, avg_imp),
                key=lambda x: x[1],
                reverse=True
            ))
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'prediction': prediction_label,
            'confidence': round(float(np.max(probabilities) * 100), 2),
            'probabilities': {
                'Normal': float(probabilities[0]) * 100,
                'Suspect': float(probabilities[1]) * 100,
                'Pathological': float(probabilities[2]) * 100
            },
            'feature_importance': feature_importance,
            'inference_time_ms': round(inference_time, 2),
            'model_used': model_name
        }

# Page config
st.set_page_config(
    page_title="Fetal Health Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #0f172a;
        color: #f1f5f9;
    }
    
    .main p, .main span, .main div:not([data-testid="stSidebar"] div), 
    .main label, .stMarkdown {
        color: #f1f5f9 !important;
    }
    
    h1, h2, h3 {
        color: #f1f5f9 !important;
    }
    
    h1 {
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.75rem;
    }
    
    [data-testid="stSidebar"] {
        background: #1e293b;
        border-right: 1px solid #475569;
    }
    
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    .stButton > button {
        background: #3b82f6 !important;
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button:hover {
        background: #2563eb !important;
        transform: translateY(-1px);
    }
    
    .status-normal {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%) !important;
        color: white !important;
        border-left: 5px solid #10b981 !important;
    }
    
    .status-suspect {
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%) !important;
        color: white !important;
        border-left: 5px solid #f59e0b !important;
    }
    
    .status-pathological {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%) !important;
        color: white !important;
        border-left: 5px solid #ef4444 !important;
    }
    
    .status-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    }
    
    .status-card h3, .status-card p {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state"""
    if 'patient_history' not in st.session_state:
        st.session_state.patient_history = []
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

# Main app
def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5rem;">🏥 Fetal Health Monitoring System</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">Fast Ensemble ML-Powered CTG Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ System Controls")
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        model_options = {
            'Ensemble Voting (Best)': 'voting_ensemble',
            'Random Forest': 'random_forest',
            'Gradient Boosting': 'gradient_boosting'
        }
        selected_model_name = st.selectbox(
            "Choose Model",
            options=list(model_options.keys())
        )
        selected_model = model_options[selected_model_name]
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📍 Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Home", "🔬 Prediction", "📜 History", "📖 Help"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model management
        st.markdown("### 🔧 Model Management")
        
        if st.button("🚀 Initialize Models", use_container_width=True):
            with st.spinner("Initializing..."):
                try:
                    manager = FetalHealthModelManager()
                    
                    # Try loading first
                    if manager.models_exist():
                        loaded = manager.load_all_models()
                        if loaded:
                            st.session_state.model_manager = manager
                            st.session_state.models_loaded = True
                            st.success("✅ Models loaded!")
                        else:
                            st.info("Training new models...")
                            manager.train_all_models()
                            st.session_state.model_manager = manager
                            st.session_state.models_loaded = True
                            st.success("✅ Models trained!")
                    else:
                        st.info("Training new models (1-2 minutes)...")
                        manager.train_all_models()
                        st.session_state.model_manager = manager
                        st.session_state.models_loaded = True
                        st.success("✅ Models ready!")
                    
                except FileNotFoundError as e:
                    st.error(f"❌ {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if st.session_state.models_loaded:
            st.success("✅ Models Ready")
        
        st.markdown("---")
        st.metric("Predictions Made", st.session_state.prediction_count)
    
    # Route to pages
    if page == "🏠 Home":
        render_home()
    elif page == "🔬 Prediction":
        render_prediction(selected_model)
    elif page == "📜 History":
        render_history()
    elif page == "📖 Help":
        render_help()

def render_home():
    """Home page"""
    st.markdown("## 🏠 Welcome")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🎯 High Accuracy**\n\nEnsemble models achieve ≥93% accuracy")
    with col2:
        st.info("**⚡ Fast Inference**\n\nPredictions in < 100ms")
    with col3:
        st.info("**🤖 Smart Ensemble**\n\nCombines RF + GB models")
    
    st.markdown("---")
    
    if st.session_state.models_loaded and st.session_state.model_manager:
        manager = st.session_state.model_manager
        
        if manager.model_metadata and 'results' in manager.model_metadata:
            st.markdown("### 📊 Model Performance")
            
            model_data = []
            for model_name, metrics in manager.model_metadata['results'].items():
                model_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.2%}"
                })
            
            df = pd.DataFrame(model_data)
            st.dataframe(df, width=800)
    else:
        st.info("👈 Initialize models to see performance metrics")

@track_performance
def render_prediction(selected_model):
    """Prediction page"""
    st.markdown("## 🔬 Fetal Health Prediction")
    
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please initialize models first!")
        return
    
    # Patient info
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", "FH-2024-001")
        gestational_age = st.number_input("Gestational Age (weeks)", 20, 42, 32)
    with col2:
        maternal_age = st.number_input("Maternal Age (years)", 18, 50, 28)
    
    st.markdown("---")
    st.markdown("### 📊 CTG Parameters")
    
    # Tabs for parameters
    tab1, tab2, tab3, tab4 = st.tabs([
        "💓 Heart Rate", 
        "📈 Accelerations", 
        "🔄 Variability", 
        "📊 Histogram"
    ])
    
    features = {}
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            features['baseline value'] = st.number_input(
                "Baseline FHR (bpm)", 100.0, 200.0, 120.0, 1.0
            )
        with col2:
            features['abnormal_short_term_variability'] = st.number_input(
                "Abnormal STV (%)", 0.0, 100.0, 50.0, 1.0
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            features['accelerations'] = st.number_input(
                "Accelerations (/s)", 0.0, 0.02, 0.003, 0.001, format="%.4f"
            )
            features['light_decelerations'] = st.number_input(
                "light decelerations (/s)", 0.0, 0.01, 0.0, 0.001, format="%.4f"
            )
        
            features['prolongued_decelerations'] = st.number_input(
                "Prolonged Decelerations (/s)", 0.0, 0.01, 0.0, 0.001, format="%.4f"
            )
        with col2:
            features['fetal_movement'] = st.number_input(
                "Fetal Movement (/s)", 0.0, 0.5, 0.0, 0.01, format="%.3f"
            )
            features['severe_decelerations'] = st.number_input(
                "Severe decelerations (/s)", 0.0, 0.01, 0.0, 0.001, format="%.4f"
            )
            features['uterine_contractions'] = st.number_input(
                "Uterine Contractions (/s)", 0.0, 0.02, 0.006, 0.001, format="%.4f"
            )
           
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            features['percentage_of_time_with_abnormal_long_term_variability'] = st.number_input(
                "% Abnormal LTV", 0.0, 100.0, 0.0, 1.0
            )
            features['mean_value_of_short_term_variability'] = st.number_input(
                "Mean STV", 0.0, 10.0, 1.0, 0.1
            )
        with col2:
            features['mean_value_of_long_term_variability'] = st.number_input(
                "Mean LTV", 0.0, 50.0, 8.0, 0.5
            )
    
    with tab4:
        col1, col2, col3 = st.columns(3)
        with col1:
            features['histogram_width'] = st.number_input("Width", 0.0, 200.0, 70.0, 1.0)
            features['histogram_min'] = st.number_input("Min", 0.0, 200.0, 60.0, 1.0)
            features['histogram_max'] = st.number_input("Max", 0.0, 300.0, 150.0, 1.0)
        with col2:
            features['histogram_number_of_peaks'] = st.number_input("Peaks", 0.0, 20.0, 3.0, 1.0)
            features['histogram_number_of_zeroes'] = st.number_input("Zeros", 0.0, 20.0, 0.0, 1.0)
            features['histogram_mode'] = st.number_input("Mode", 0.0, 200.0, 120.0, 1.0)
        with col3:
            features['histogram_mean'] = st.number_input("Mean", 0.0, 200.0, 130.0, 1.0)
            features['histogram_median'] = st.number_input("Median", 0.0, 200.0, 130.0, 1.0)
            features['histogram_variance'] = st.number_input("Variance", 0.0, 300.0, 20.0, 1.0)
            features['histogram_tendency'] = st.number_input("Tendency", -1.0, 1.0, 0.0, 1.0)
    
    st.markdown("---")
    
    if st.button("🔍 Analyze", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            try:
                manager = st.session_state.model_manager
                result = manager.predict(features, selected_model)
                
                display_results(result)
                save_to_history(patient_id, gestational_age, maternal_age, result)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def display_results(result):
    """Display prediction results"""
    prediction = result['prediction']
    confidence = result['confidence']
    
    status_class = f"status-{prediction.lower()}"
    icon = {"Normal": "✅", "Suspect": "⚠️", "Pathological": "🚨"}[prediction]
    
    st.markdown(f"""
    <div class="status-card {status_class}">
        <h3>{icon} Prediction: {prediction}</h3>
        <p><strong>Confidence:</strong> {confidence}%</p>
        <p><strong>Inference Time:</strong> {result['inference_time_ms']}ms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probabilities
    st.markdown("### 📊 Class Probabilities")
    
    classes = list(result['probabilities'].keys())
    probs = list(result['probabilities'].values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            marker_color=['#10b981', '#f59e0b', '#ef4444'],
            text=[f"{p:.1f}%" for p in probs],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font=dict(color='#f1f5f9'),
        height=400
    )
    st.plotly_chart(fig, width='stretch')
    
    # Recommendations
    st.markdown("### 🩺 Recommendations")
    if prediction == "Normal":
        st.success("**Reassuring Pattern** - Continue routine monitoring")
    elif prediction == "Suspect":
        st.warning("**Borderline Findings** - Repeat CTG within 24 hours, consider specialist consultation")
    else:
        st.error("**Concerning Pattern** - URGENT: Notify physician immediately, prepare for intervention")

def save_to_history(patient_id, gestational_age, maternal_age, result):
    """Save to history"""
    record = {
        'timestamp': datetime.now(),
        'patient_id': patient_id,
        'gestational_age': gestational_age,
        'maternal_age': maternal_age,
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'model_used': result['model_used']
    }
    
    st.session_state.patient_history.append(record)
    st.session_state.prediction_count += 1

def render_history():
    """History page"""
    st.markdown("## 📜 Prediction History")
    
    if not st.session_state.patient_history:
        st.info("No predictions yet")
        return
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    normal = sum(1 for r in st.session_state.patient_history if r['prediction'] == 'Normal')
    suspect = sum(1 for r in st.session_state.patient_history if r['prediction'] == 'Suspect')
    path = sum(1 for r in st.session_state.patient_history if r['prediction'] == 'Pathological')
    
    col1.metric("Total", st.session_state.prediction_count)
    col2.metric("🟢 Normal", normal)
    col3.metric("🟡 Suspect", suspect)
    col4.metric("🔴 Pathological", path)
    
    # Table
    df = pd.DataFrame([{
        'Time': r['timestamp'].strftime('%Y-%m-%d %H:%M'),
        'Patient ID': r['patient_id'],
        'Prediction': r['prediction'],
        'Confidence': f"{r['confidence']}%"
    } for r in st.session_state.patient_history])
    
    st.dataframe(df, width='stretch')

def render_help():
    """Help page"""
    st.markdown("## 📖 Help & Documentation")
    
    with st.expander("🎯 About"):
        st.markdown("""
        Fast ensemble ML system for fetal health classification using CTG data.
        
        **Features:**
        - 3 ML models (RF, GB, Ensemble)
        - ≥93% accuracy
        - < 100ms inference
        """)
    
    with st.expander("⚠️ Disclaimer"):
        st.warning("""
        **For demonstration and research only**
        - Not FDA approved
        - Not for clinical decisions
        - Always consult healthcare professionals
        """)

if __name__ == "__main__":
    main()
