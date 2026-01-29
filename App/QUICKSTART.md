# 🚀 Quick Start Guide

Get your Fetal Health Monitoring System up and running in 5 minutes!

## ⚡ Quick Setup (Local)

### Step 1: Install Python
Make sure you have Python 3.8 or higher installed:
```bash
python --version
```

### Step 2: Download the Files
Download all project files to a folder on your computer.

### Step 3: Install Dependencies
```bash
cd path/to/fetal_health_app
pip install -r requirements.txt
```

### Step 4: Run the App
```bash
streamlit run app.py
```

### Step 5: Open in Browser
The app will automatically open at `http://localhost:8501`

**That's it! You're ready to use the system. 🎉**

## 🌐 Quick Deploy (Streamlit Cloud)

### 1. Upload to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR-USERNAME/fetal-health-monitoring.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository and `app.py`
5. Click "Deploy"

**Your app is live in 2-3 minutes! 🌍**

## 📝 First Use Guide

### Using Quick Test Mode (Easiest)
1. **Open the app**
2. **In the sidebar**: Select "Quick Test" from Input Mode
3. **Choose a test case**: "Normal - Healthy Fetus"
4. **Click**: "🔍 Analyze CTG Data"
5. **View results**: See prediction, confidence, and recommendations

### Using Manual Entry
1. **In the sidebar**: Select "Manual Entry"
2. **Enter patient info**: ID, gestational age, maternal age
3. **Fill in CTG data**: Expand each section and enter values
4. **Click**: "🔍 Analyze CTG Data"
5. **Review results**: Check prediction and clinical recommendations

### Using CSV Upload
1. **In the sidebar**: Select "Upload CSV"
2. **Download template**: Click "📥 Download CSV Template"
3. **Fill in data**: Open CSV and enter your measurements
4. **Upload file**: Select your completed CSV
5. **Click**: "🔍 Analyze CTG Data"

## 🎯 Key Features

| Feature | Description | Location |
|---------|-------------|----------|
| Patient Info | Enter patient details | Sidebar |
| Input Mode | Choose how to input data | Sidebar |
| New Analysis | Make predictions | Tab 1 |
| History | View past predictions | Tab 2 |
| Help | Documentation & guides | Tab 3 |
| Export | Download reports | After prediction |

## 💡 Tips for Best Results

1. **Use Quick Test first** - Familiarize yourself with the interface
2. **Check reference ranges** - Hover over input fields for guidance
3. **Review all tabs** - Explore History and Help sections
4. **Export reports** - Download JSON reports for record-keeping
5. **Monitor trends** - Use History tab to track patterns

## ⚠️ Important Notes

- This is a **demonstration system** for educational purposes
- Not approved for clinical use without validation
- Always use clinical judgment with AI predictions
- System uses mock predictor (not real ML models in this version)

## 🐛 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| App won't start | Run `pip install -r requirements.txt` |
| Import errors | Check Python version (3.8+ required) |
| Charts not showing | Clear browser cache or try different browser |
| Upload fails | Check CSV file format matches template |

## 📚 Next Steps

After getting familiar with the basics:

1. **Explore different test cases** - Try all three scenarios
2. **Test CSV upload** - Practice with the template
3. **Review documentation** - Check the Help tab
4. **Track history** - Make multiple predictions to see trends
5. **Customize settings** - Toggle display options in sidebar

## 🎓 Learning Resources

- **User Guide**: See the Help & Documentation tab in the app
- **README.md**: Full project documentation
- **DEPLOYMENT.md**: Detailed deployment instructions
- **Streamlit Docs**: https://docs.streamlit.io

## 📞 Need Help?

- Check the **Help tab** in the application
- Review the **troubleshooting** section above
- Consult **README.md** for detailed information
- Visit **Streamlit Community**: https://discuss.streamlit.io

---

**Enjoy using the Fetal Health Monitoring System! 🏥**

*If you found this helpful, give the project a ⭐ on GitHub!*
