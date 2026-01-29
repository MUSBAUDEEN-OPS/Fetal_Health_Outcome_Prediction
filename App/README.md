# 🏥 Fetal Health Monitoring System

A production-ready Streamlit web application for real-time fetal health classification using AI-powered CTG (Cardiotocography) analysis.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/streamlit-1.31%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Overview

The Fetal Health Monitoring System is an AI-powered clinical decision support tool designed to assist healthcare professionals in assessing fetal well-being through Cardiotocography (CTG) analysis. The system provides:

- 🔬 **Real-time CTG Analysis**: Instant AI-powered predictions
- 📊 **Interactive Visualizations**: Clear, professional charts and graphs
- 📈 **Trend Tracking**: Monitor patient history over time
- 💾 **Data Export**: Download reports and history in CSV/JSON format
- 🎯 **Three Input Modes**: Manual entry, CSV upload, or quick test cases

## 🚀 Features

### Core Functionality
- **Three Classification Categories**:
  - Normal: Healthy fetal status
  - Suspect: Borderline findings requiring monitoring
  - Pathological: Concerning patterns requiring immediate attention

- **21 CTG Features Analyzed**:
  - Baseline heart rate
  - Accelerations and decelerations
  - Heart rate variability (short-term and long-term)
  - Histogram features and statistical analysis

- **Clinical Decision Support**:
  - Confidence scores
  - Probability distributions
  - Evidence-based recommendations
  - Feature importance analysis

### User Interface
- Clean, medical-professional aesthetic
- Responsive design for desktop and tablet
- Dark sidebar with light main content
- Interactive Plotly charts
- Real-time session statistics

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fetal-health-monitoring.git
cd fetal-health-monitoring
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the application**
Open your browser and navigate to `http://localhost:8501`

## 🌐 Deployment on Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** and push your code:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/fetal-health-monitoring.git
git push -u origin main
```

2. **Ensure these files are in your repository**:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `.streamlit/config.toml` (configuration)
   - `README.md` (documentation)

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch (main), and main file (app.py)
5. Click "Deploy"

Your app will be live at: `https://[your-app-name].streamlit.app`

### Step 3: Configure Secrets (if needed)

If you have API keys or sensitive data:
1. Go to your app settings on Streamlit Cloud
2. Click "Secrets"
3. Add your secrets in TOML format

## 📖 Usage Guide

### Manual Entry Mode
1. Select "Manual Entry" in the sidebar
2. Enter patient information (ID, gestational age, maternal age)
3. Fill in all 21 CTG measurements in the organized form
4. Click "🔍 Analyze CTG Data"
5. Review results, recommendations, and feature importance

### CSV Upload Mode
1. Select "Upload CSV" in the sidebar
2. Download the CSV template
3. Fill in your data following the template format
4. Upload the completed CSV file
5. Click "🔍 Analyze CTG Data"

### Quick Test Mode
1. Select "Quick Test" in the sidebar
2. Choose from pre-configured test cases:
   - Normal - Healthy Fetus
   - Suspect - Borderline
   - Pathological - Concerning
3. Click "🔍 Analyze CTG Data"
4. See instant results

## 📊 Understanding the Results

### Prediction Categories

**Normal (Green)**
- Healthy fetal status
- Reassuring heart rate patterns
- Continue routine monitoring

**Suspect (Yellow)**
- Borderline findings
- Requires increased surveillance
- Consider repeat CTG

**Pathological (Red)**
- Concerning patterns
- Immediate clinical attention required
- May require urgent intervention

### Confidence Score
- High (≥70%): Strong confidence in prediction
- Low (<70%): Uncertainty present, additional assessment recommended

## 🔧 Configuration

### Streamlit Configuration (.streamlit/config.toml)

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#f5f7fa"
secondaryBackgroundColor = "#ffffff"
textColor = "#1a202c"

[server]
maxUploadSize = 200
enableCORS = false
```

### Display Options (Sidebar)
- Show Feature Importance: Toggle feature analysis chart
- Show Probability Distribution: Toggle probability bar chart
- Show Detailed Metrics: Toggle additional patient metrics

## 🗂️ Project Structure

```
fetal-health-monitoring/
│
├── app.py                      # Main application file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── .streamlit/
│   └── config.toml            # Streamlit configuration
│
└── .gitignore                 # Git ignore file
```

## 📋 Requirements

```
streamlit>=1.31.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
openpyxl>=3.1.0
xlsxwriter>=3.1.0
python-dateutil>=2.8.0
```

## 🔒 Important Disclaimers

⚠️ **This system is a demonstration/decision support tool**

- For educational and demonstration purposes only
- Not approved for clinical use without proper validation
- Always use clinical judgment in conjunction with AI predictions
- Does not replace expert obstetric consultation
- Requires regulatory approval before clinical deployment

## 📱 Browser Compatibility

Tested and optimized for:
- ✅ Google Chrome (recommended)
- ✅ Mozilla Firefox
- ✅ Microsoft Edge
- ✅ Safari

## 🐛 Troubleshooting

### Common Issues

**Issue**: App won't start
```bash
# Solution: Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

**Issue**: Import errors
```bash
# Solution: Check Python version
python --version  # Should be 3.8+
```

**Issue**: Charts not displaying
```bash
# Solution: Clear Streamlit cache
streamlit cache clear
```

### Getting Help

If you encounter issues:
1. Check the [Streamlit documentation](https://docs.streamlit.io)
2. Review the troubleshooting section above
3. Check your browser console for errors
4. Ensure all dependencies are correctly installed

## 🔄 Version History

### v1.0.0 (January 2026)
- ✅ Initial production release
- ✅ Three input modes (Manual, CSV, Quick Test)
- ✅ Mock predictor for demonstration
- ✅ Real-time prediction capabilities
- ✅ Patient history tracking
- ✅ Export and reporting features
- ✅ Professional medical UI/UX
- ✅ Interactive visualizations
- ✅ Clinical recommendations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions, suggestions, or support:
- **Email**: support@fetalhealthai.com
- **Documentation**: www.fetalhealthai.com/docs
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/fetal-health-monitoring/issues)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Visualizations powered by [Plotly](https://plotly.com)
- Data handling with [Pandas](https://pandas.pydata.org)
- Designed for healthcare professionals worldwide

---

**⚕️ Healthcare AI Solutions © 2026**

*Empowering healthcare professionals with AI-driven insights*
