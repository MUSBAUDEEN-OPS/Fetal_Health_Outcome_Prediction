# Project Structure Guide

This document explains how to organize the Fetal Health Monitoring System files for optimal functionality.

## Recommended Folder Structure

Your project folder should be organized as follows:

```
fetal_health_monitoring/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Complete documentation
├── DEPLOYMENT.md                   # Deployment instructions
├── QUICKSTART.md                   # Quick start guide
│
├── .streamlit/                     # Streamlit configuration (optional)
│   └── config.toml                # Configuration settings
│
├── venv/                          # Virtual environment (created by you)
│   ├── Scripts/ (Windows)         # or bin/ (macOS/Linux)
│   └── ...
│
├── models/                        # Machine learning models (optional)
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── scaler.pkl
│
├── data/                          # Sample data files (optional)
│   ├── sample_ctg_data.csv
│   └── test_cases.json
│
└── logs/                          # Application logs (created automatically)
    └── fetal_monitor.log
```

## Essential Files

These files are required for the application to run:

### app.py
The main application file containing all the code for the Streamlit interface, machine learning predictions, and user interactions. This is the file you run with the `streamlit run app.py` command.

### requirements.txt
A text file listing all the Python packages needed by the application. When you run `pip install -r requirements.txt`, Python reads this file and installs everything listed in it.

## Documentation Files

These files provide information and instructions but aren't required for the app to run:

### README.md
The comprehensive documentation covering everything about the system, including features, usage instructions, technical details, and important disclaimers. This is your go-to resource for detailed information.

### DEPLOYMENT.md
A detailed guide for deploying the application in various environments, from local development to cloud deployment. Consult this when you want to make the app accessible to others.

### QUICKSTART.md
A condensed guide to get you up and running as quickly as possible. Perfect for first-time users who want to see the app in action without reading extensive documentation.

## Optional Directories

### .streamlit/
This hidden folder contains configuration files for Streamlit. The most important file is `config.toml`, which allows you to customize various aspects of the application such as the port number, theme colors, and server settings.

You can create this folder if you want to customize the application's behavior:
- Windows: Create a folder named `.streamlit` in your project directory
- macOS/Linux: The folder is already hidden because it starts with a dot

### models/
If you train your own machine learning models or want to use pre-trained models, store them in this directory. The application can be modified to load models from this location rather than using the built-in demo prediction logic.

In a production environment, this folder would contain:
- Trained model files (saved using pickle or joblib)
- The scaler object used to normalize input features
- Model metadata and version information

### data/
This optional folder can hold sample datasets, test cases, or training data. It's useful if you want to:
- Store example CSV files for testing the upload functionality
- Keep reference datasets for validation
- Maintain pre-configured test scenarios

### logs/
If you enable logging in the application (which is recommended for production use), log files will be stored here. These logs help you:
- Track application usage and performance
- Debug issues that occur during operation
- Maintain an audit trail of predictions made
- Monitor system health over time

## Setting Up the Structure

### Quick Setup (Minimal)

For basic usage, you only need these files in one folder:
1. app.py
2. requirements.txt

Create a project folder and place both files there. That's all you need to get started with the Quick Start guide.

### Complete Setup (Recommended)

For a more organized project, create the following structure:

**Step 1:** Create the main project folder
```bash
mkdir fetal_health_monitoring
cd fetal_health_monitoring
```

**Step 2:** Place all the essential files in this folder
- app.py
- requirements.txt
- README.md
- DEPLOYMENT.md
- QUICKSTART.md

**Step 3:** Create optional directories
```bash
mkdir models data logs
```

**Step 4:** Create the .streamlit configuration folder (optional)
```bash
mkdir .streamlit
```

Then place the config.toml file inside the .streamlit folder if you want custom configuration.

**Step 5:** Create and activate your virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 6:** Install dependencies
```bash
pip install -r requirements.txt
```

Now your project is fully set up and ready to use!

## File Locations Based on Use Case

### For Learning and Testing
You only need the essential files (app.py and requirements.txt) in one folder. The documentation files are helpful but not required for the app to function.

### For Local Clinical Use
Include the essential files plus the .streamlit configuration folder. This allows you to customize the interface and configure network access settings.

### For Network Deployment
Use the complete structure including the models folder if you're using custom trained models. Keep logs enabled to track usage and performance.

### For Production Deployment
Use the complete structure with all directories. Additionally, implement proper security measures, backups, and monitoring as described in the DEPLOYMENT.md guide.

## Important Notes

### Virtual Environment (venv/)
The virtual environment folder (venv) should NOT be included when sharing or distributing your project. Each user should create their own virtual environment on their system. This folder is specific to your computer and Python installation.

If you're using version control (like Git), add venv/ to your .gitignore file to prevent it from being committed.

### Data Privacy
If you're working with real patient data, make sure any CSV files or logs containing sensitive information are properly secured and not shared or committed to version control. The data/ and logs/ folders should be excluded from any public repositories.

### Customization
Feel free to modify this structure to fit your needs. The application is flexible and can work with different organizational approaches as long as the essential files (app.py and requirements.txt) are present.

## Checking Your Structure

To verify your project is set up correctly, your project folder should contain at minimum:

1. ✓ app.py (the main application file)
2. ✓ requirements.txt (the dependencies file)
3. ✓ venv/ folder (after you've created the virtual environment)

If you have these three items, you're ready to run the application!

To run a quick check, open a terminal in your project folder and run:
- Windows: `dir` to list all files and folders
- macOS/Linux: `ls -la` to list all files and folders (including hidden ones)

You should see app.py and requirements.txt listed among the files.

## Moving Forward

Once you have the basic structure set up, follow the QUICKSTART.md guide to get the application running, then explore the README.md for comprehensive information about all features and capabilities.

If you plan to deploy the application for others to use, consult DEPLOYMENT.md for detailed instructions on various deployment scenarios.

---

This structure is designed to keep your project organized, maintainable, and ready for both development and production use. Start simple with just the essential files and expand as your needs grow!
