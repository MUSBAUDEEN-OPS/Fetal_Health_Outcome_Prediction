# Quick Start Guide - Fetal Health Monitoring System

Get up and running in 5 minutes!

## For First-Time Users

### Step 1: Install Python (if not already installed)

**Check if you have Python:**
Open your terminal or command prompt and type:
```bash
python --version
```

If you see a version number (3.8 or higher), you're good to go! Skip to Step 2.

If not, download Python from python.org and install it. During installation, make sure to check "Add Python to PATH".

### Step 2: Download the Project Files

Make sure you have all these files in one folder:
- app.py (the main application)
- requirements.txt (list of dependencies)
- README.md (full documentation)
- DEPLOYMENT.md (deployment guide)

### Step 3: Open Terminal in the Project Folder

**Windows:**
- Navigate to the folder in File Explorer
- Hold Shift and right-click in the folder
- Select "Open PowerShell window here" or "Open command window here"

**macOS:**
- Navigate to the folder in Finder
- Right-click the folder and select "New Terminal at Folder"

**Linux:**
- Navigate to the folder in your file manager
- Right-click and select "Open in Terminal"

### Step 4: Set Up Virtual Environment

This creates an isolated space for the project's dependencies.

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your command line. This means the virtual environment is active.

### Step 5: Install Dependencies

With the virtual environment activated, install all required packages:

**Windows:**
```cmd
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
pip install -r requirements.txt
```

This will take a few minutes as it downloads and installs all the necessary libraries. You'll see progress messages as each package is installed.

### Step 6: Launch the Application

Start the Streamlit server:

**Windows:**
```cmd
streamlit run app.py
```

**macOS/Linux:**
```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`. If it doesn't open automatically, just copy and paste that URL into your browser.

## Using the Application

Once the application is running, you'll see a professional medical interface. Here's how to get started:

### Your First Prediction

1. **Enter Patient Information** (left sidebar):
   - Patient ID: Enter any identifier (e.g., "TEST001")
   - Gestational Age: Enter a value between 20-42 weeks
   - Maternal Age: Enter the mother's age

2. **Choose Quick Test Mode** (easiest way to start):
   - In the sidebar, select "Quick Test" under Input Mode
   - Choose "Normal - Healthy Fetus" from the dropdown
   - This loads pre-configured test data

3. **Analyze the Data**:
   - Scroll down and click the blue "Analyze CTG Data" button
   - Wait a moment while the system processes the data

4. **Review the Results**:
   - You'll see a prediction (Normal, Suspect, or Pathological)
   - Confidence score showing how certain the model is
   - A probability distribution chart
   - Clinical recommendations

### Try Different Scenarios

Experiment with the other test cases to see how the system handles different situations:
- "Suspect - Borderline Case" shows a case needing increased monitoring
- "Pathological - High Risk" demonstrates concerning patterns

### Manual Data Entry

Once you're comfortable with the Quick Test mode, try Manual Entry:
1. Select "Manual Entry" in the sidebar
2. Fill in the CTG measurements in the organized forms
3. Hover over the information icons (ⓘ) to see normal ranges
4. Click "Analyze CTG Data" when ready

## Stopping the Application

When you're done:
1. Go to the terminal window where Streamlit is running
2. Press `Ctrl+C` (on both Windows and macOS/Linux)
3. The server will stop and you can close the terminal

To deactivate the virtual environment (optional):
```bash
deactivate
```

## Running the Application Again

The next time you want to use the application:

1. Open terminal in the project folder
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
3. Run the app: `streamlit run app.py`

You don't need to reinstall the dependencies unless you've deleted the venv folder.

## Common First-Time Issues

### "Python is not recognized"
**Solution:** You need to install Python or add it to your PATH. Restart your computer after installing Python.

### "No module named 'streamlit'"
**Solution:** Make sure your virtual environment is activated (you should see `(venv)` in your terminal). Then run `pip install -r requirements.txt` again.

### Port Already in Use
**Solution:** Another program is using port 8501. Either close that program or run Streamlit on a different port:
```bash
streamlit run app.py --server.port 8502
```

### Application Won't Open in Browser
**Solution:** Manually type `http://localhost:8501` into your web browser's address bar.

## Next Steps

Once you're comfortable with the basics:

1. **Read the Full Documentation**: Check out README.md for detailed information about all features
2. **Try CSV Upload**: Create a CSV file with CTG data for batch processing
3. **Explore History Tab**: See how the system tracks predictions over time
4. **Review Deployment Guide**: If you want to share the app with others on your network, see DEPLOYMENT.md

## Getting Help

If you run into issues:
1. Check the terminal output for error messages
2. Read the relevant section in README.md
3. Review the troubleshooting section in DEPLOYMENT.md
4. Make sure all files are in the same folder
5. Verify your Python version is 3.8 or higher

## Important Reminders

- This is a demonstration system for educational purposes
- Always use clinical judgment in conjunction with AI predictions
- The system is designed to support, not replace, healthcare professionals
- For clinical deployment, additional validation and regulatory approval is required

---

Congratulations! You now have a working Fetal Health Monitoring System. Explore the interface, try different features, and refer to the comprehensive documentation in README.md for more advanced usage.

**Enjoy exploring the system!**
