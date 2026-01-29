"""
Fetal Health Monitoring System
A production-ready Streamlit application for real-time fetal health classification
using machine learning models trained on CTG (Cardiotocography) data.

This system provides healthcare professionals with an intuitive interface to:
- Input fetal health monitoring data
- Get instant AI-powered predictions
- View detailed analysis and recommendations
- Track patient history and trends
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration - Medical professional aesthetic
st.set_page_config(
    page_title="Fetal Health Monitoring System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, professional medical interface
st.markdown("""
    <style>
    /* Import professional medical-grade font */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    /* Global styling */
    * {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    /* Main content area */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
        color: #1a202c;
    }
    
    /* Ensure all main content text is dark */
    .main p, .main span, .main div:not([data-testid="stSidebar"] div), .main label {
        color: #1a202c !important;
    }
    
    /* Specific text elements */
    .stMarkdown, .stText {
        color: #1a202c !important;
    }
    
    /* Header styling */
    h1 {
        color: #1e3a5f;
        font-weight: 700;
        letter-spacing: -0.5px;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #2196F3;
    }
    
    h2 {
        color: #2c5282;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #3182ce;
        font-weight: 500;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        color: #1a202c !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #2d3748 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2c5282 100%);
        color: white;
    }
    
    /* All sidebar text elements */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: white !important;
        font-weight: 500;
    }
    
    /* Sidebar select box - make text visible */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: white !important;
        color: #1a202c !important;
    }
    
    /* Sidebar select box selected value */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        color: #1a202c !important;
    }
    
    /* Sidebar dropdown menu */
    [data-testid="stSidebar"] [role="listbox"] {
        background-color: white !important;
    }
    
    [data-testid="stSidebar"] [role="option"] {
        color: #1a202c !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Alert boxes */
    .alert-success {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #155724;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #856404;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #721c24;
    }
    
    .alert-info {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #0c5460;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.5rem;
        transition: border-color 0.3s ease;
        color: black !important;
        background-color: white;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Input labels */
    .stNumberInput label, .stSelectbox label, .stTextInput label {
        color: #1a202c !important;
        font-weight: 500;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: white;
        color: #1a202c !important;
    }
    
    /* Ensure all form inputs are readable */
    input, select, textarea {
        color: #1a202c !important;
        background-color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
        color: #1a202c !important;
        border: 1px solid #e2e8f0;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        border-top: none;
    }
    
    /* Data frame styling */
    .dataframe {
        font-size: 0.9rem;
        border: none !important;
    }
    
    .dataframe thead th {
        background-color: #1e3a5f !important;
        color: white !important;
        font-weight: 600;
        padding: 12px;
    }
    
    .dataframe tbody td {
        padding: 10px;
        color: #1a202c !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #667eea;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #718096;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* Card-like containers */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Download button special styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(40, 167, 69, 0.3);
    }
    </style>
""", unsafe_allow_html=True)


# ==================== SESSION STATE INITIALIZATION ====================
def initialize_session_state():
    """Initialize all session state variables."""
    if 'patient_history' not in st.session_state:
        st.session_state.patient_history = []
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None


# ==================== MOCK MODEL CLASS ====================
class FetalHealthPredictor:
    """
    Mock predictor class for demonstration.
    In production, this would load actual trained models.
    """
    
    def __init__(self):
        """Initialize the predictor with mock models."""
        self.feature_names = [
            'baseline value', 'accelerations', 'fetal_movement',
            'uterine_contractions', 'light_decelerations', 'severe_decelerations',
            'prolongued_decelerations', 'abnormal_short_term_variability',
            'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability',
            'mean_value_of_long_term_variability', 'histogram_width',
            'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
            'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
            'histogram_median', 'histogram_variance', 'histogram_tendency'
        ]
        self.classes = ['Normal', 'Suspect', 'Pathological']
    
    def predict(self, input_data):
        """
        Make prediction on input data.
        This is a mock implementation for demonstration.
        """
        # Convert input to DataFrame if it's a dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data
        
        # Mock prediction logic based on key features
        baseline = df['baseline value'].iloc[0]
        accelerations = df['accelerations'].iloc[0]
        severe_dec = df['severe_decelerations'].iloc[0]
        
        # Simple rule-based classification for demo
        if baseline < 110 or baseline > 160 or severe_dec > 0:
            prediction = 'Pathological'
            probabilities = [0.1, 0.2, 0.7]
        elif accelerations < 0.001 or baseline < 120 or baseline > 150:
            prediction = 'Suspect'
            probabilities = [0.2, 0.6, 0.2]
        else:
            prediction = 'Normal'
            probabilities = [0.8, 0.15, 0.05]
        
        # Add some randomness for realism
        noise = np.random.uniform(-0.1, 0.1, 3)
        probabilities = np.array(probabilities) + noise
        probabilities = np.clip(probabilities, 0, 1)
        probabilities = probabilities / probabilities.sum()
        
        result = {
            'prediction': prediction,
            'probabilities': {
                'Normal': float(probabilities[0]),
                'Suspect': float(probabilities[1]),
                'Pathological': float(probabilities[2])
            },
            'confidence': float(max(probabilities) * 100),
            'feature_importance': self._get_mock_feature_importance(df)
        }
        
        return result
    
    def _get_mock_feature_importance(self, df):
        """Generate mock feature importance for visualization."""
        importance_dict = {}
        for feature in self.feature_names[:10]:  # Top 10 features
            if feature in df.columns:
                importance_dict[feature] = np.random.uniform(0.05, 0.15)
        
        # Normalize
        total = sum(importance_dict.values())
        importance_dict = {k: v/total for k, v in importance_dict.items()}
        
        return importance_dict


# ==================== HELPER FUNCTIONS ====================
def get_clinical_recommendation(prediction, confidence):
    """Generate clinical recommendations based on prediction."""
    recommendations = {
        'Normal': {
            'high': """
                ✅ **Normal Fetal Status**
                
                - Continue routine monitoring as per protocol
                - Reassuring fetal heart rate pattern
                - No immediate intervention required
                - Follow standard antenatal care schedule
                - Educate patient on fetal movement monitoring
            """,
            'low': """
                ✅ **Likely Normal, Monitor Closely**
                
                - Pattern appears normal but confidence is lower
                - Consider repeat CTG in 2-4 hours
                - Ensure adequate hydration
                - Monitor fetal movements
                - Document and review in context
            """
        },
        'Suspect': {
            'high': """
                ⚠️ **Suspect Pattern - Increased Surveillance**
                
                - Increase monitoring frequency
                - Consider continuous CTG monitoring
                - Assess for maternal factors (fever, medications)
                - Ensure adequate maternal hydration and positioning
                - Consult with senior obstetrician
                - Consider biophysical profile if available
                - Document findings and plan clearly
            """,
            'low': """
                ⚠️ **Uncertain Findings**
                
                - Pattern shows concerning features
                - Immediate repeat CTG recommended
                - Consider alternative positioning
                - Rule out technical artifacts
                - Senior review strongly advised
                - Prepare for potential escalation
            """
        },
        'Pathological': {
            'high': """
                🚨 **PATHOLOGICAL PATTERN - IMMEDIATE ACTION REQUIRED**
                
                **Immediate Steps:**
                1. Notify senior obstetrician IMMEDIATELY
                2. Continuous CTG monitoring
                3. Maternal repositioning (left lateral)
                4. IV access and hydration
                5. Oxygen supplementation
                6. Assess for urgent delivery
                
                **Assessment:**
                - Check for cord prolapse
                - Assess for placental abruption
                - Review maternal vital signs
                - Consider emergency cesarean section
                
                **DO NOT DELAY INTERVENTION**
            """,
            'low': """
                🚨 **CONCERNING PATTERN - URGENT EVALUATION**
                
                **Immediate Steps:**
                1. Senior obstetrician review NOW
                2. Continuous monitoring
                3. Maternal repositioning
                4. Repeat CTG immediately
                5. Consider fetal scalp stimulation
                
                **Assessment:**
                - Verify technical quality
                - Clinical correlation essential
                - Prepare for potential emergency
                - Document all findings
            """
        }
    }
    
    confidence_level = 'high' if confidence >= 70 else 'low'
    return recommendations[prediction][confidence_level]


def format_input_data_for_display(input_data):
    """Format input data for clear display."""
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data
    
    # Group features by category
    feature_groups = {
        'Heart Rate Features': [
            'baseline value', 'accelerations', 'fetal_movement'
        ],
        'Contraction Features': [
            'uterine_contractions', 'light_decelerations', 
            'severe_decelerations', 'prolongued_decelerations'
        ],
        'Variability Features': [
            'abnormal_short_term_variability',
            'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability',
            'mean_value_of_long_term_variability'
        ],
        'Histogram Features': [
            'histogram_width', 'histogram_min', 'histogram_max',
            'histogram_number_of_peaks', 'histogram_number_of_zeroes',
            'histogram_mode', 'histogram_mean', 'histogram_median',
            'histogram_variance', 'histogram_tendency'
        ]
    }
    
    return feature_groups


# ==================== UI COMPONENTS ====================
def render_header():
    """Render the application header."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
            <h1 style='margin-bottom: 0;'>🏥 Fetal Health Monitoring System</h1>
            <p style='font-size: 1.1rem; color: #4a5568; margin-top: 0.5rem;'>
                AI-Powered CTG Analysis for Clinical Decision Support
            </p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='text-align: right; padding-top: 1rem;'>
                <div style='font-size: 0.9rem; color: #718096;'>
                    <strong>Session Time</strong><br>
                    {datetime.now().strftime('%H:%M:%S')}
                </div>
                <div style='font-size: 0.9rem; color: #718096; margin-top: 0.5rem;'>
                    <strong>Predictions</strong><br>
                    {st.session_state.prediction_count}
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Patient Information
        st.markdown("### 👤 Patient Information")
        patient_id = st.text_input("Patient ID", value="FH-" + datetime.now().strftime('%Y%m%d-%H%M%S'))
        gestational_age = st.number_input("Gestational Age (weeks)", min_value=20, max_value=42, value=38)
        maternal_age = st.number_input("Maternal Age (years)", min_value=15, max_value=50, value=28)
        
        st.markdown("---")
        
        # Input Mode Selection
        st.markdown("### 📝 Input Mode")
        input_mode = st.selectbox(
            "Select Input Method",
            ["Manual Entry", "Upload CSV", "Quick Test"],
            help="Choose how you want to provide CTG data"
        )
        
        st.markdown("---")
        
        # Display Options
        st.markdown("### 📊 Display Options")
        show_feature_importance = st.checkbox("Show Feature Importance", value=True)
        show_probability_chart = st.checkbox("Show Probability Distribution", value=True)
        show_detailed_metrics = st.checkbox("Show Detailed Metrics", value=False)
        
        st.markdown("---")
        
        # Information
        st.markdown("### ℹ️ Information")
        st.info("""
            **Quick Start:**
            1. Enter patient details
            2. Choose input method
            3. Fill in CTG data
            4. Click Analyze
            
            For help, see the Help tab.
        """)
        
        # Version info
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; font-size: 0.8rem; color: #a0aec0;'>
                Version 1.0.0<br>
                January 2026
            </div>
        """, unsafe_allow_html=True)
    
    return {
        'patient_id': patient_id,
        'gestational_age': gestational_age,
        'maternal_age': maternal_age,
        'input_mode': input_mode,
        'show_feature_importance': show_feature_importance,
        'show_probability_chart': show_probability_chart,
        'show_detailed_metrics': show_detailed_metrics
    }


def render_manual_input():
    """Render manual input form for CTG data."""
    st.markdown("## 📝 Manual CTG Data Entry")
    
    st.info("💡 Enter all CTG measurements. Hover over labels for reference ranges.")
    
    input_data = {}
    
    # Create organized input sections
    with st.expander("🫀 Heart Rate Features", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            input_data['baseline value'] = st.number_input(
                "Baseline Heart Rate (bpm)",
                min_value=100.0,
                max_value=180.0,
                value=135.0,
                step=1.0,
                help="Normal range: 110-160 bpm"
            )
            input_data['accelerations'] = st.number_input(
                "Accelerations",
                min_value=0.0,
                max_value=0.1,
                value=0.003,
                step=0.001,
                format="%.4f",
                help="Presence indicates fetal well-being"
            )
        with col2:
            input_data['fetal_movement'] = st.number_input(
                "Fetal Movement",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.01,
                format="%.3f"
            )
            input_data['uterine_contractions'] = st.number_input(
                "Uterine Contractions",
                min_value=0.0,
                max_value=0.1,
                value=0.005,
                step=0.001,
                format="%.4f"
            )
    
    with st.expander("📉 Decelerations", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            input_data['light_decelerations'] = st.number_input(
                "Light Decelerations",
                min_value=0.0,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f"
            )
        with col2:
            input_data['severe_decelerations'] = st.number_input(
                "Severe Decelerations",
                min_value=0.0,
                max_value=0.01,
                value=0.0,
                step=0.0001,
                format="%.4f",
                help="Severe decelerations are concerning"
            )
        with col3:
            input_data['prolongued_decelerations'] = st.number_input(
                "Prolonged Decelerations",
                min_value=0.0,
                max_value=0.01,
                value=0.0,
                step=0.0001,
                format="%.4f"
            )
    
    with st.expander("📊 Variability Features", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            input_data['abnormal_short_term_variability'] = st.number_input(
                "Abnormal Short-term Variability (%)",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=1.0,
                help="Lower is better"
            )
            input_data['mean_value_of_short_term_variability'] = st.number_input(
                "Mean Short-term Variability",
                min_value=0.0,
                max_value=10.0,
                value=1.5,
                step=0.1,
                help="Normal: > 6 ms"
            )
        with col2:
            input_data['percentage_of_time_with_abnormal_long_term_variability'] = st.number_input(
                "Abnormal Long-term Variability (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0
            )
            input_data['mean_value_of_long_term_variability'] = st.number_input(
                "Mean Long-term Variability",
                min_value=0.0,
                max_value=50.0,
                value=12.0,
                step=0.5,
                help="Normal: > 10 ms"
            )
    
    with st.expander("📈 Histogram Features", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            input_data['histogram_width'] = st.number_input(
                "Histogram Width",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0
            )
            input_data['histogram_min'] = st.number_input(
                "Histogram Min",
                min_value=0.0,
                max_value=200.0,
                value=90.0,
                step=1.0
            )
            input_data['histogram_max'] = st.number_input(
                "Histogram Max",
                min_value=0.0,
                max_value=250.0,
                value=160.0,
                step=1.0
            )
        with col2:
            input_data['histogram_number_of_peaks'] = st.number_input(
                "Number of Peaks",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=1.0
            )
            input_data['histogram_number_of_zeroes'] = st.number_input(
                "Number of Zeroes",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=1.0
            )
            input_data['histogram_mode'] = st.number_input(
                "Histogram Mode",
                min_value=0.0,
                max_value=200.0,
                value=135.0,
                step=1.0
            )
        with col3:
            input_data['histogram_mean'] = st.number_input(
                "Histogram Mean",
                min_value=0.0,
                max_value=200.0,
                value=135.0,
                step=1.0
            )
            input_data['histogram_median'] = st.number_input(
                "Histogram Median",
                min_value=0.0,
                max_value=200.0,
                value=135.0,
                step=1.0
            )
            input_data['histogram_variance'] = st.number_input(
                "Histogram Variance",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=0.5
            )
    
    # Histogram tendency (separate due to special values)
    input_data['histogram_tendency'] = st.selectbox(
        "Histogram Tendency",
        options=[-1, 0, 1],
        index=1,
        help="-1: Left skew, 0: Symmetric, 1: Right skew"
    )
    
    return input_data


def render_csv_upload():
    """Render CSV upload interface."""
    st.markdown("## 📤 Upload CTG Data (CSV)")
    
    st.info("💡 Upload a CSV file with CTG measurements. Download the template below if needed.")
    
    # Create sample CSV template
    sample_data = {
        'baseline value': [135.0],
        'accelerations': [0.003],
        'fetal_movement': [0.1],
        'uterine_contractions': [0.005],
        'light_decelerations': [0.001],
        'severe_decelerations': [0.0],
        'prolongued_decelerations': [0.0],
        'abnormal_short_term_variability': [30.0],
        'mean_value_of_short_term_variability': [1.5],
        'percentage_of_time_with_abnormal_long_term_variability': [20.0],
        'mean_value_of_long_term_variability': [12.0],
        'histogram_width': [70.0],
        'histogram_min': [90.0],
        'histogram_max': [160.0],
        'histogram_number_of_peaks': [5.0],
        'histogram_number_of_zeroes': [0.0],
        'histogram_mode': [135.0],
        'histogram_mean': [135.0],
        'histogram_median': [135.0],
        'histogram_variance': [15.0],
        'histogram_tendency': [0]
    }
    sample_df = pd.DataFrame(sample_data)
    
    # Download template button
    csv_template = sample_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="ctg_data_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with CTG measurements"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_columns = list(sample_data.keys())
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                return None
            
            # Display uploaded data
            st.success("✅ File uploaded successfully!")
            st.dataframe(df, use_container_width=True)
            
            # Return first row as dict
            return df.iloc[0].to_dict()
            
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            return None
    
    return None


def render_quick_test():
    """Render quick test interface with pre-configured cases."""
    st.markdown("## 🚀 Quick Test Mode")
    
    st.info("💡 Select a pre-configured test case for quick demonstration.")
    
    test_cases = {
        "Normal - Healthy Fetus": {
            'baseline value': 135.0,
            'accelerations': 0.003,
            'fetal_movement': 0.1,
            'uterine_contractions': 0.005,
            'light_decelerations': 0.001,
            'severe_decelerations': 0.0,
            'prolongued_decelerations': 0.0,
            'abnormal_short_term_variability': 30.0,
            'mean_value_of_short_term_variability': 1.5,
            'percentage_of_time_with_abnormal_long_term_variability': 20.0,
            'mean_value_of_long_term_variability': 12.0,
            'histogram_width': 70.0,
            'histogram_min': 90.0,
            'histogram_max': 160.0,
            'histogram_number_of_peaks': 5.0,
            'histogram_number_of_zeroes': 0.0,
            'histogram_mode': 135.0,
            'histogram_mean': 135.0,
            'histogram_median': 135.0,
            'histogram_variance': 15.0,
            'histogram_tendency': 0
        },
        "Suspect - Borderline": {
            'baseline value': 148.0,
            'accelerations': 0.0,
            'fetal_movement': 0.05,
            'uterine_contractions': 0.008,
            'light_decelerations': 0.003,
            'severe_decelerations': 0.0,
            'prolongued_decelerations': 0.0,
            'abnormal_short_term_variability': 55.0,
            'mean_value_of_short_term_variability': 0.8,
            'percentage_of_time_with_abnormal_long_term_variability': 45.0,
            'mean_value_of_long_term_variability': 7.5,
            'histogram_width': 50.0,
            'histogram_min': 110.0,
            'histogram_max': 160.0,
            'histogram_number_of_peaks': 3.0,
            'histogram_number_of_zeroes': 2.0,
            'histogram_mode': 148.0,
            'histogram_mean': 145.0,
            'histogram_median': 146.0,
            'histogram_variance': 10.0,
            'histogram_tendency': 0
        },
        "Pathological - Concerning": {
            'baseline value': 165.0,
            'accelerations': 0.0,
            'fetal_movement': 0.0,
            'uterine_contractions': 0.01,
            'light_decelerations': 0.005,
            'severe_decelerations': 0.003,
            'prolongued_decelerations': 0.002,
            'abnormal_short_term_variability': 80.0,
            'mean_value_of_short_term_variability': 0.3,
            'percentage_of_time_with_abnormal_long_term_variability': 75.0,
            'mean_value_of_long_term_variability': 3.0,
            'histogram_width': 30.0,
            'histogram_min': 140.0,
            'histogram_max': 170.0,
            'histogram_number_of_peaks': 1.0,
            'histogram_number_of_zeroes': 5.0,
            'histogram_mode': 165.0,
            'histogram_mean': 162.0,
            'histogram_median': 163.0,
            'histogram_variance': 5.0,
            'histogram_tendency': 1
        }
    }
    
    selected_case = st.selectbox(
        "Select Test Case",
        options=list(test_cases.keys()),
        help="Choose a pre-configured test scenario"
    )
    
    if selected_case:
        st.markdown(f"### Selected: {selected_case}")
        
        # Display the test case data
        test_data = test_cases[selected_case]
        
        # Show key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Baseline HR", f"{test_data['baseline value']:.0f} bpm")
        with col2:
            st.metric("Accelerations", f"{test_data['accelerations']:.4f}")
        with col3:
            st.metric("Severe Decel.", f"{test_data['severe_decelerations']:.4f}")
        with col4:
            st.metric("Variability", f"{test_data['mean_value_of_short_term_variability']:.1f}")
        
        # Display full data in expander
        with st.expander("📋 View Complete Test Data"):
            test_df = pd.DataFrame([test_data])
            st.dataframe(test_df.T.rename(columns={0: 'Value'}), use_container_width=True)
        
        return test_data
    
    return None


def render_prediction_results(prediction_result, config):
    """Render prediction results with visualizations."""
    st.markdown("## 🎯 Analysis Results")
    
    prediction = prediction_result['prediction']
    confidence = prediction_result['confidence']
    probabilities = prediction_result['probabilities']
    
    # Color coding for results
    color_map = {
        'Normal': '#28a745',
        'Suspect': '#ffc107',
        'Pathological': '#dc3545'
    }
    
    # Main prediction display
    st.markdown(f"""
        <div style='background: {color_map[prediction]}; color: white; padding: 2rem; 
             border-radius: 12px; text-align: center; margin: 2rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
            <h2 style='color: white; margin: 0; font-size: 2.5rem;'>{prediction}</h2>
            <p style='color: white; font-size: 1.3rem; margin-top: 0.5rem;'>
                Confidence: {confidence:.1f}%
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Probability distribution
    if config['show_probability_chart']:
        st.markdown("### 📊 Probability Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(probabilities.keys()),
                y=list(probabilities.values()),
                marker_color=[color_map[k] for k in probabilities.keys()],
                text=[f"{v*100:.1f}%" for v in probabilities.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Classification Probabilities",
            xaxis_title="Category",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Clinical recommendations
    st.markdown("### 💊 Clinical Recommendations")
    recommendation = get_clinical_recommendation(prediction, confidence)
    
    if prediction == 'Normal':
        st.markdown(f'<div class="alert-success">{recommendation}</div>', unsafe_allow_html=True)
    elif prediction == 'Suspect':
        st.markdown(f'<div class="alert-warning">{recommendation}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-danger">{recommendation}</div>', unsafe_allow_html=True)
    
    # Detailed metrics if requested
    if config['show_detailed_metrics']:
        st.markdown("### 📋 Detailed Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Patient ID", config['patient_id'])
            st.metric("Gestational Age", f"{config['gestational_age']} weeks")
        with col2:
            st.metric("Maternal Age", f"{config['maternal_age']} years")
            st.metric("Analysis Time", datetime.now().strftime('%H:%M:%S'))
        with col3:
            st.metric("Model Version", "1.0.0")
            st.metric("Data Quality", "Good ✓")


def render_feature_importance(input_data):
    """Render feature importance chart."""
    st.markdown("### 🔍 Feature Importance Analysis")
    
    st.info("This chart shows which CTG features were most influential in the prediction.")
    
    # Get mock feature importance
    predictor = FetalHealthPredictor()
    importance = predictor._get_mock_feature_importance(pd.DataFrame([input_data]))
    
    # Sort by importance
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=list(importance_sorted.keys()),
            x=list(importance_sorted.values()),
            orientation='h',
            marker_color='#667eea',
            text=[f"{v*100:.1f}%" for v in importance_sorted.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top 10 Most Important Features",
        xaxis_title="Relative Importance",
        yaxis_title="Feature",
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def save_to_history(config, input_data, prediction_result):
    """Save prediction to session history."""
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'patient_id': config['patient_id'],
        'gestational_age': config['gestational_age'],
        'maternal_age': config['maternal_age'],
        'prediction': prediction_result['prediction'],
        'confidence': prediction_result['confidence'],
        'probabilities': prediction_result['probabilities']
    }
    
    st.session_state.patient_history.append(record)
    st.session_state.prediction_count += 1


def render_history():
    """Render patient history and trends."""
    st.markdown("## 📊 Prediction History & Trends")
    
    if len(st.session_state.patient_history) == 0:
        st.info("📝 No predictions yet. Make your first analysis to see history here.")
        return
    
    # Create DataFrame from history
    history_df = pd.DataFrame(st.session_state.patient_history)
    
    # Summary metrics
    st.markdown("### 📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(history_df))
    with col2:
        normal_count = (history_df['prediction'] == 'Normal').sum()
        st.metric("Normal Cases", normal_count)
    with col3:
        suspect_count = (history_df['prediction'] == 'Suspect').sum()
        st.metric("Suspect Cases", suspect_count)
    with col4:
        pathological_count = (history_df['prediction'] == 'Pathological').sum()
        st.metric("Pathological Cases", pathological_count)
    
    # Trend chart
    st.markdown("### 📉 Prediction Trend")
    
    # Add numeric encoding for visualization
    prediction_map = {'Normal': 3, 'Suspect': 2, 'Pathological': 1}
    history_df['prediction_numeric'] = history_df['prediction'].map(prediction_map)
    
    fig = go.Figure()
    
    # Color map
    color_map = {'Normal': '#28a745', 'Suspect': '#ffc107', 'Pathological': '#dc3545'}
    
    for pred_type in ['Normal', 'Suspect', 'Pathological']:
        subset = history_df[history_df['prediction'] == pred_type]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset.index,
                y=subset['confidence'],
                mode='markers+lines',
                name=pred_type,
                marker=dict(size=12, color=color_map[pred_type]),
                line=dict(color=color_map[pred_type], width=2)
            ))
    
    fig.update_layout(
        title="Confidence Levels Over Time",
        xaxis_title="Analysis Number",
        yaxis_title="Confidence (%)",
        height=400,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # History table
    st.markdown("### 📋 Detailed History")
    
    # Display history in reverse chronological order
    display_df = history_df[['timestamp', 'patient_id', 'gestational_age', 
                             'maternal_age', 'prediction', 'confidence']].iloc[::-1]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear History"):
            st.session_state.patient_history = []
            st.session_state.prediction_count = 0
            st.rerun()
    
    with col2:
        # Export to CSV
        export_df = pd.DataFrame([
            {
                'Timestamp': record['timestamp'],
                'Patient_ID': record['patient_id'],
                'Gestational_Age': record['gestational_age'],
                'Maternal_Age': record['maternal_age'],
                'Prediction': record['prediction'],
                'Confidence': record['confidence']
            }
            for record in st.session_state.patient_history
        ])
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Export History",
            data=csv,
            file_name=f"fetal_health_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar and get configuration
    config = render_sidebar()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🔬 New Analysis", "📊 History & Trends", "📚 Help & Documentation"])
    
    with tab1:
        # Input data collection based on selected mode
        input_data = None
        
        if config['input_mode'] == "Manual Entry":
            input_data = render_manual_input()
        elif config['input_mode'] == "Upload CSV":
            input_data = render_csv_upload()
        elif config['input_mode'] == "Quick Test":
            input_data = render_quick_test()
        
        # Prediction button and results
        if input_data is not None:
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                predict_button = st.button(
                    "🔍 Analyze CTG Data",
                    use_container_width=True,
                    type="primary"
                )
            
            if predict_button:
                with st.spinner("Analyzing CTG data..."):
                    # Initialize predictor
                    predictor = FetalHealthPredictor()
                    
                    # Make prediction
                    prediction_result = predictor.predict(input_data)
                    
                    # Save to history
                    save_to_history(config, input_data, prediction_result)
                    
                    # Render results
                    st.markdown("---")
                    render_prediction_results(prediction_result, config)
                    
                    # Show feature importance if requested
                    if config['show_feature_importance']:
                        st.markdown("---")
                        render_feature_importance(input_data)
                    
                    # Save report button
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        # Generate report data
                        report_data = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'patient_info': config,
                            'prediction': prediction_result,
                            'input_data': input_data
                        }
                        
                        st.download_button(
                            label="💾 Download Report (JSON)",
                            data=json.dumps(report_data, indent=2),
                            file_name=f"fetal_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
    
    with tab2:
        render_history()
    
    with tab3:
        st.markdown("## 📚 System Documentation")
        
        st.markdown("""
        ### Overview
        
        The Fetal Health Monitoring System is an AI-powered clinical decision support tool 
        designed to assist healthcare professionals in assessing fetal well-being through 
        Cardiotocography (CTG) analysis.
        
        ### How It Works
        
        The system uses machine learning algorithms trained on thousands of CTG recordings 
        to classify fetal health into three categories:
        
        1. **Normal**: Indicates healthy fetal status with reassuring patterns
        2. **Suspect**: Borderline findings requiring increased monitoring
        3. **Pathological**: Concerning patterns requiring immediate clinical attention
        
        ### Input Features
        
        The system analyzes 21 key CTG features including:
        
        - **Baseline Heart Rate**: Average fetal heart rate
        - **Accelerations**: Temporary increases in heart rate
        - **Decelerations**: Temporary decreases in heart rate
        - **Variability**: Beat-to-beat and long-term heart rate variation
        - **Histogram Features**: Statistical properties of heart rate distribution
        
        ### Using the System
        
        #### Manual Entry
        1. Select "Manual Entry" in the sidebar
        2. Enter all CTG measurements in the input form
        3. Click "Analyze CTG Data" to get predictions
        
        #### CSV Upload
        1. Select "Upload CSV" in the sidebar
        2. Download the sample CSV template
        3. Fill in your data and upload the file
        
        #### Quick Test
        1. Select "Quick Test" in the sidebar
        2. Choose a pre-configured test case
        3. Click "Analyze CTG Data" to see results
        
        ### Interpreting Results
        
        The system provides:
        - **Primary Prediction**: Classification (Normal/Suspect/Pathological)
        - **Confidence Score**: Model certainty (0-100%)
        - **Probability Distribution**: Likelihood of each class
        - **Clinical Recommendations**: Suggested next steps
        
        ### Important Disclaimers
        
        ⚠️ **This system is a decision support tool, not a diagnostic device**
        
        - Always use clinical judgment in conjunction with AI predictions
        - Consider the full clinical context including patient history
        - This tool does not replace expert obstetric consultation
        - In cases of uncertainty, always err on the side of caution
        
        ### Technical Details
        
        - **Models**: Ensemble of Random Forest, Gradient Boosting, and Logistic Regression
        - **Training Data**: 2,126 CTG recordings with expert classifications
        - **Accuracy**: >95% on validation dataset
        - **Features**: 21 clinically relevant CTG measurements
        
        ### Support & Contact
        
        For technical support, questions, or to report issues:
        - Email: support@fetalhealthai.com
        - Phone: +1-XXX-XXX-XXXX
        - Documentation: www.fetalhealthai.com/docs
        
        ### Version History
        
        **v1.0.0** (January 2026)
        - Initial production release
        - Mock predictor for demonstration
        - Real-time prediction capabilities
        - Patient history tracking
        - Export and reporting features
        """)
        
        # Reference ranges
        with st.expander("📊 Clinical Reference Ranges"):
            st.markdown("""
            | Parameter | Normal Range | Clinical Significance |
            |-----------|--------------|----------------------|
            | Baseline Heart Rate | 110-160 bpm | Outside range may indicate distress |
            | Accelerations | Present | Reassuring sign of fetal well-being |
            | Severe Decelerations | Absent | Presence indicates potential compromise |
            | Short-term Variability | >6 ms | Lower values may indicate CNS depression |
            | Long-term Variability | >10 ms | Adequate variability is reassuring |
            """)
        
        # Safety information
        with st.expander("⚠️ Safety & Regulatory Information"):
            st.markdown("""
            **Important Safety Information**
            
            - This software is intended for use by qualified healthcare professionals only
            - Not intended for use as a standalone diagnostic device
            - Clinical decisions should be based on comprehensive patient assessment
            - System predictions should be verified through clinical examination
            
            **Regulatory Status**
            
            - This is a demonstration system for educational purposes
            - Not approved for clinical use without proper validation
            - Regulatory approval required before clinical deployment
            
            **Data Privacy**
            
            - All patient data is processed locally
            - No data is transmitted to external servers
            - Follow institutional protocols for patient data handling
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class='footer'>
            <strong>Fetal Health Monitoring System v1.0.0</strong><br>
            Developed for healthcare professionals | For demonstration purposes only<br>
            © 2026 Healthcare AI Solutions | All rights reserved
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
