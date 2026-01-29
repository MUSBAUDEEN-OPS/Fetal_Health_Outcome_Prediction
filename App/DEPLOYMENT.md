# 🚀 Deployment Guide for GitHub and Streamlit Cloud

This guide will walk you through deploying your Fetal Health Monitoring System on Streamlit Cloud using GitHub.

## 📋 Prerequisites

Before you begin, ensure you have:
- ✅ A GitHub account ([sign up here](https://github.com/join))
- ✅ Git installed on your computer ([download here](https://git-scm.com/downloads))
- ✅ All project files in a local folder

## 🔄 Step 1: Create a GitHub Repository

### Option A: Using GitHub Web Interface

1. **Go to GitHub** and sign in
2. **Click the "+" icon** in the top right corner
3. **Select "New repository"**
4. **Configure your repository**:
   - Repository name: `fetal-health-monitoring` (or your preferred name)
   - Description: "AI-powered fetal health monitoring system using CTG analysis"
   - Visibility: Public (required for free Streamlit Cloud deployment)
   - ✅ Add a README file (you can replace it later)
   - Choose a license: MIT License (recommended)
5. **Click "Create repository"**

### Option B: Using Command Line

```bash
# Navigate to your project folder
cd /path/to/fetal_health_app

# Initialize git repository
git init

# Add all files
git add .

# Commit files
git commit -m "Initial commit: Fetal Health Monitoring System"

# Create repository on GitHub (you'll need to do this via web first)
# Then link your local repository
git remote add origin https://github.com/YOUR-USERNAME/fetal-health-monitoring.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 📤 Step 2: Upload Your Files to GitHub

If you created the repository via web interface:

```bash
# Clone your repository
git clone https://github.com/YOUR-USERNAME/fetal-health-monitoring.git

# Copy your project files into the cloned folder
# Then:
cd fetal-health-monitoring

# Add all files
git add .

# Commit
git commit -m "Add application files"

# Push to GitHub
git push origin main
```

## ✅ Step 3: Verify Your Repository Structure

Your GitHub repository should contain:
```
fetal-health-monitoring/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

**Important**: Make sure all these files are visible on GitHub!

## 🌐 Step 4: Deploy on Streamlit Cloud

### 4.1 Sign Up for Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Sign up with GitHub"
3. Authorize Streamlit to access your GitHub account
4. Complete your profile setup

### 4.2 Deploy Your App

1. **Click "New app"** button (top right)
2. **Configure deployment**:
   - **Repository**: Select `YOUR-USERNAME/fetal-health-monitoring`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom URL (e.g., `fetal-health-monitor`)
3. **Advanced settings** (optional):
   - **Python version**: 3.11 (recommended)
   - **Secrets**: Leave empty (unless you have API keys)
4. **Click "Deploy"**

### 4.3 Monitor Deployment

- Streamlit Cloud will install dependencies from `requirements.txt`
- You'll see real-time logs during deployment
- Deployment typically takes 2-5 minutes
- Once complete, your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

## 🎉 Step 5: Access Your Deployed App

Your app is now live! You can:
- Share the URL with others
- Access it from any device
- Make updates by pushing to GitHub (auto-deploys)

**Your app URL**: `https://YOUR-APP-NAME.streamlit.app`

## 🔧 Step 6: Managing Your Deployment

### Updating Your App

When you make changes:
```bash
# Make your changes to the code
# Then:
git add .
git commit -m "Description of changes"
git push origin main
```

Streamlit Cloud will automatically detect changes and redeploy your app!

### Viewing Logs

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click on your app
3. View real-time logs and metrics
4. Check resource usage

### Restarting Your App

If your app needs a restart:
1. Go to your app management page
2. Click the "⋮" menu (three dots)
3. Select "Reboot app"

## 📊 Step 7: Monitor App Performance

Streamlit Cloud provides:
- **Resource usage**: CPU, memory, and bandwidth
- **Visitor statistics**: Page views and unique visitors
- **Error logs**: Debugging information
- **App status**: Uptime monitoring

Access these via your app's management dashboard.

## 🔒 Best Practices

### Security
- ✅ Never commit API keys or passwords to GitHub
- ✅ Use Streamlit secrets for sensitive data
- ✅ Keep your repository public for free hosting
- ✅ Review code before pushing to main branch

### Performance
- ✅ Use `@st.cache_data` for expensive computations
- ✅ Optimize image sizes
- ✅ Minimize external API calls
- ✅ Use session state efficiently

### Maintenance
- ✅ Keep dependencies updated
- ✅ Monitor error logs regularly
- ✅ Test locally before deploying
- ✅ Use meaningful commit messages

## 🐛 Troubleshooting Common Issues

### Issue 1: App Won't Deploy

**Symptoms**: Deployment fails with errors

**Solutions**:
```bash
# Check requirements.txt is correct
cat requirements.txt

# Ensure all imports in app.py are in requirements.txt
# Test locally first:
pip install -r requirements.txt
streamlit run app.py
```

### Issue 2: Module Not Found Error

**Symptoms**: "ModuleNotFoundError: No module named 'xxx'"

**Solution**: Add missing package to `requirements.txt`
```txt
# Add the missing package
package-name>=version
```

### Issue 3: App Crashes on Startup

**Symptoms**: App shows error page immediately

**Solutions**:
1. Check Streamlit Cloud logs for error messages
2. Verify all file paths are relative, not absolute
3. Ensure no hardcoded local paths
4. Test with minimal code first

### Issue 4: Slow Performance

**Solutions**:
1. Use `@st.cache_data` decorator for expensive functions
2. Reduce image sizes
3. Optimize data loading
4. Consider Streamlit Cloud Pro for more resources

### Issue 5: GitHub Push Rejected

**Symptoms**: "error: failed to push some refs"

**Solution**:
```bash
# Pull latest changes first
git pull origin main

# Then push
git push origin main
```

## 📱 Mobile Optimization

Your app is automatically responsive, but for better mobile experience:

1. Test on mobile devices
2. Use Streamlit's built-in responsive design
3. Avoid very wide charts (use `use_container_width=True`)
4. Keep forms concise

## 🔄 Continuous Integration (Optional)

For advanced users, set up GitHub Actions:

1. Create `.github/workflows/test.yml`
2. Add automated testing
3. Run checks before deployment

Example workflow:
```yaml
name: Streamlit App CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Test app
        run: |
          python -c "import streamlit"
```

## 📈 Scaling Your Application

### Free Tier Limits
- 1 GB RAM
- 1 CPU core
- 1 GB storage
- Public apps only

### Upgrading to Pro
If you need more resources:
1. Go to Streamlit Cloud settings
2. Upgrade to Pro plan
3. Get access to:
   - More resources
   - Private apps
   - Custom domains
   - Priority support

## 🎓 Additional Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Streamlit Community**: https://discuss.streamlit.io
- **GitHub Docs**: https://docs.github.com
- **Streamlit Gallery**: https://streamlit.io/gallery

## 📞 Getting Help

If you encounter issues:

1. **Check Streamlit Cloud logs** - Often shows the exact error
2. **Search Streamlit Community** - Many issues already solved
3. **GitHub Issues** - Check your repository's issues
4. **Streamlit Support** - For Pro users

## ✅ Deployment Checklist

Before deploying, ensure:

- [ ] All files committed to GitHub
- [ ] `requirements.txt` is complete and correct
- [ ] App runs locally without errors
- [ ] No hardcoded file paths
- [ ] No sensitive data in code
- [ ] `.streamlit/config.toml` present
- [ ] `.gitignore` properly configured
- [ ] README.md is informative
- [ ] Repository is public (for free tier)
- [ ] All dependencies compatible with Python 3.8+

## 🎉 You're Done!

Your Fetal Health Monitoring System is now deployed and accessible worldwide!

**Next Steps**:
- Share your app URL
- Monitor usage and feedback
- Iterate and improve
- Consider adding real ML models

---

**Need help?** Check the troubleshooting section or contact support.

**Happy deploying! 🚀**
