# Troubleshooting Common Errors - Fetal Health Monitoring System

This guide addresses common errors you might encounter when deploying the Fetal Health Monitoring System, with specific solutions for each.

## Table of Contents
1. [Plotly ValueError with update_layout](#plotly-valueerror-with-update_layout)
2. [Module Import Errors](#module-import-errors)
3. [Port Binding Errors](#port-binding-errors)
4. [Memory Issues](#memory-issues)
5. [Session State Errors](#session-state-errors)

---

## Plotly ValueError with update_layout

### Error Message
```
ValueError: This app has encountered an error
File "app.py", line 1112, in render_prediction_results
    fig.update_layout(...)
```

### Cause
This error occurs due to compatibility issues between Plotly versions and how nested dictionaries are handled in the `update_layout()` method. Python 3.13 and newer Plotly versions have stricter requirements for dictionary formatting.

### Solution
The latest version of `app.py` (provided with this fix) has been updated to use dictionary literals `{}` instead of `dict()` constructors in Plotly calls. If you're still experiencing this issue:

1. **Update to the latest app.py file** - The fixed version has already been provided
2. **Verify your Plotly version**:
   ```bash
   pip show plotly
   ```
   Should be version 5.18.0 or compatible

3. **If using an older version of the code**, look for these patterns and replace them:
   
   **OLD (causes error):**
   ```python
   fig.update_layout(
       title=dict(
           text='My Title',
           font=dict(size=18)
       ),
       xaxis=dict(title='X Axis'),
       margin=dict(t=80, b=60)
   )
   ```
   
   **NEW (fixed):**
   ```python
   fig.update_layout(
       title={'text': 'My Title', 'font': {'size': 18}},
       xaxis={'title': 'X Axis'},
       margin={'t': 80, 'b': 60}
   )
   ```

### Prevention
- Use the updated requirements.txt with version constraints
- Test the application in a fresh virtual environment before deploying
- Always use dictionary literals `{}` instead of `dict()` in Plotly configurations

---

## Module Import Errors

### Error Message
```
ModuleNotFoundError: No module named 'streamlit'
ModuleNotFoundError: No module named 'plotly'
```

### Cause
Required packages are not installed in the current Python environment.

### Solution

1. **Verify you're in the virtual environment**:
   - You should see `(venv)` at the start of your terminal prompt
   - If not, activate it:
     ```bash
     # Windows
     venv\Scripts\activate
     
     # macOS/Linux
     source venv/bin/activate
     ```

2. **Install all requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **If installation fails**, try upgrading pip first:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **For specific package issues**, install them individually:
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn matplotlib seaborn
   ```

### For Streamlit Cloud Deployment
Ensure your requirements.txt file is in the root directory of your repository and properly formatted. Streamlit Cloud automatically installs packages from this file during deployment.

---

## Port Binding Errors

### Error Message
```
OSError: [Errno 98] Address already in use
StreamlitAddressInUseException: Port 8501 is already in use
```

### Cause
Another application or previous instance of Streamlit is using port 8501.

### Solution

**Option 1: Stop the conflicting process**

On **Linux/macOS**:
```bash
# Find the process using port 8501
lsof -i :8501

# Kill the process (replace PID with the actual process ID)
kill -9 PID
```

On **Windows**:
```cmd
# Find the process using port 8501
netstat -ano | findstr :8501

# Kill the process (replace PID with the actual process ID)
taskkill /PID PID /F
```

**Option 2: Use a different port**
```bash
streamlit run app.py --server.port 8502
```

**Option 3: Configure default port**
Create/modify `.streamlit/config.toml`:
```toml
[server]
port = 8502
```

---

## Memory Issues

### Error Message
```
MemoryError
RuntimeError: out of memory
```

### Cause
The application or machine learning models are consuming too much memory, especially on machines with limited RAM or when processing large datasets.

### Solution

1. **Close other applications** to free up memory

2. **Restart the Streamlit server**:
   - Press Ctrl+C to stop
   - Run again: `streamlit run app.py`

3. **For local deployment**, ensure you have at least:
   - 2GB RAM for basic usage
   - 4GB RAM recommended for smooth operation

4. **For cloud deployment**, upgrade to a larger instance:
   - Streamlit Cloud: Use a paid tier for more resources
   - AWS/GCP/Azure: Select instance with at least 2GB RAM (e.g., t2.small or larger)

5. **Optimize the application**:
   - Add caching to expensive operations
   - Clear session state when not needed:
     ```python
     if st.button("Clear Session"):
         st.session_state.clear()
         st.rerun()
     ```

---

## Session State Errors

### Error Message
```
StreamlitSessionStateError
KeyError: 'patient_history'
AttributeError: 'st.session_state' has no attribute 'patient_history'
```

### Cause
Session state variables are accessed before being initialized, or the session was reset unexpectedly.

### Solution

The application includes proper initialization in the main code:
```python
if 'patient_history' not in st.session_state:
    st.session_state.patient_history = []
```

If you're still experiencing issues:

1. **Clear browser cache and cookies** for localhost:8501

2. **Force reload the page**:
   - Chrome/Edge: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
   - Firefox: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)

3. **Restart the Streamlit server**:
   - Stop the server (Ctrl+C)
   - Clear Python cache: `find . -type d -name __pycache__ -exec rm -rf {} +` (Linux/Mac)
   - Start again: `streamlit run app.py`

4. **Add defensive checks** in your code:
   ```python
   # Before accessing session state
   if 'your_variable' not in st.session_state:
       st.session_state.your_variable = default_value
   ```

---

## Other Common Issues

### Python Version Compatibility

**Error**: Various syntax or import errors

**Solution**: Ensure you're using Python 3.8 or higher:
```bash
python --version
```

If needed, install a compatible Python version from python.org.

### File Path Issues

**Error**: 
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution**:
- Verify all files are in the correct locations
- Use absolute paths or properly configure relative paths
- Ensure the working directory is set correctly:
  ```bash
  cd /path/to/fetal_health_monitoring
  streamlit run app.py
  ```

### Permission Denied Errors (Linux/macOS)

**Error**:
```
PermissionError: [Errno 13] Permission denied
```

**Solution**:
```bash
# Fix file permissions
chmod +x app.py

# Fix directory permissions
chmod -R 755 /path/to/project

# If needed, change ownership
sudo chown -R $USER:$USER /path/to/project
```

### Network Access Issues (Institutional Firewalls)

**Error**: Unable to access application from other machines

**Solution**:
1. Configure Streamlit to listen on all interfaces:
   ```toml
   # .streamlit/config.toml
   [server]
   address = "0.0.0.0"
   ```

2. Open firewall port:
   ```bash
   # Linux (ufw)
   sudo ufw allow 8501/tcp
   
   # Check if port is open
   sudo ufw status
   ```

3. Verify network connectivity:
   ```bash
   # Test from another machine
   curl http://your-server-ip:8501
   ```

---

## Getting Additional Help

If you continue experiencing issues:

1. **Check the terminal output** for detailed error messages
2. **Enable debug mode** in Streamlit:
   ```bash
   streamlit run app.py --logger.level=debug
   ```

3. **Check Streamlit logs**:
   - Location varies by OS
   - Usually in `~/.streamlit/` directory

4. **Search for similar issues**:
   - Streamlit Community Forum: discuss.streamlit.io
   - Stack Overflow: stackoverflow.com/questions/tagged/streamlit
   - GitHub Issues: github.com/streamlit/streamlit/issues

5. **Provide detailed information** when asking for help:
   - Complete error message and traceback
   - Python version: `python --version`
   - Streamlit version: `streamlit version`
   - Operating system
   - Steps to reproduce the error

---

## Deployment Checklist

Before deploying, verify:

- [ ] All requirements in requirements.txt are installed
- [ ] Virtual environment is activated
- [ ] No syntax errors in app.py
- [ ] Firewall rules configured (if deploying on network)
- [ ] Sufficient system resources (RAM, disk space)
- [ ] Python version 3.8 or higher
- [ ] All files in correct locations
- [ ] Browser cache cleared (if redeploying)
- [ ] Application runs without errors locally first

---

## Version-Specific Issues

### Streamlit 1.31+
- Uses new caching decorators (`@st.cache_data` instead of `@st.cache`)
- Session state initialization required before access

### Python 3.13
- Stricter type checking in some libraries
- Dictionary unpacking behavior changed
- Use the updated app.py provided with this fix

### Plotly 5.18+
- Changed dictionary handling in layout updates
- Prefer dictionary literals over dict() constructors
- Some deprecated parameters removed

---

This troubleshooting guide covers the most common issues. For deployment-specific guidance, refer to DEPLOYMENT.md. For general usage questions, see README.md and QUICKSTART.md.

**Last Updated**: January 2026  
**Compatible with**: Python 3.8-3.13, Streamlit 1.31+, Plotly 5.18+
