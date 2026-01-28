# Deployment Guide - Fetal Health Monitoring System

This guide provides detailed instructions for deploying the Fetal Health Monitoring System in various environments, from local development to production cloud deployment.

## Table of Contents

1. Local Development Setup
2. Network Deployment (Institutional Use)
3. Cloud Deployment (AWS/Azure/GCP)
4. Docker Containerization
5. Security Considerations
6. Troubleshooting

---

## 1. Local Development Setup

### Windows Setup

#### Step 1: Install Python
Download and install Python 3.8 or higher from python.org. During installation, make sure to check the box that says "Add Python to PATH".

#### Step 2: Verify Installation
Open Command Prompt and verify Python is installed:

```cmd
python --version
pip --version
```

You should see version numbers for both commands. If you see an error, you may need to restart your computer or manually add Python to your PATH.

#### Step 3: Create Project Directory
Create a folder for the project and navigate to it:

```cmd
mkdir fetal_health_monitoring
cd fetal_health_monitoring
```

#### Step 4: Set Up Virtual Environment
Create an isolated Python environment for the project:

```cmd
python -m venv venv
venv\Scripts\activate
```

After activation, you should see `(venv)` at the beginning of your command prompt, indicating the virtual environment is active.

#### Step 5: Install Dependencies
With the virtual environment activated, install all required packages:

```cmd
pip install -r requirements.txt
```

This process may take a few minutes as it downloads and installs all necessary libraries.

#### Step 6: Launch Application
Start the Streamlit server:

```cmd
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`. If it doesn't open automatically, you can manually navigate to this URL.

### macOS/Linux Setup

The setup process is similar to Windows with minor differences in command syntax:

#### Step 1: Verify Python Installation
Most macOS and Linux systems come with Python pre-installed. Check your version:

```bash
python3 --version
pip3 --version
```

If Python is not installed or the version is below 3.8, install it using your system's package manager (Homebrew for macOS, apt for Ubuntu, etc.).

#### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Launch Application
```bash
streamlit run app.py
```

---

## 2. Network Deployment (Institutional Use)

For deployment on an institutional network where multiple users need access from different computers, you need to configure Streamlit to accept connections from other machines on the network.

### Step 1: Configure Streamlit
Create a configuration file to allow network access. First, create the Streamlit config directory:

```bash
mkdir -p ~/.streamlit
```

Create a file named `config.toml` in the `.streamlit` directory with the following content:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 8501

[browser]
gatherUsageStats = false
```

This configuration tells Streamlit to listen on all network interfaces, making it accessible from other computers on the network.

### Step 2: Find Your Server IP Address

**Windows:**
```cmd
ipconfig
```
Look for "IPv4 Address" under your active network connection.

**macOS/Linux:**
```bash
ifconfig
```
or
```bash
ip addr show
```

Note your IP address (typically something like 192.168.1.100 for local networks).

### Step 3: Configure Firewall
You need to allow incoming connections on port 8501.

**Windows Firewall:**
1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Create a new Inbound Rule
4. Select "Port" and click Next
5. Enter port 8501
6. Allow the connection
7. Apply to all profiles and name it "Streamlit Fetal Health Monitor"

**macOS Firewall:**
```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add streamlit
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp streamlit
```

**Linux (ufw):**
```bash
sudo ufw allow 8501/tcp
```

### Step 4: Launch for Network Access
Run the application:

```bash
streamlit run app.py
```

Users on your network can now access the application by navigating to `http://YOUR-IP-ADDRESS:8501` in their web browsers, replacing YOUR-IP-ADDRESS with the IP address you found in Step 2.

### Step 5: Keep Application Running
For persistent deployment, use a process manager:

**Using screen (simple option):**
```bash
screen -S fetal_monitor
streamlit run app.py
# Press Ctrl+A, then D to detach
# Reattach with: screen -r fetal_monitor
```

**Using systemd (Linux, more robust):**
Create a service file at `/etc/systemd/system/fetal-health.service`:

```ini
[Unit]
Description=Fetal Health Monitoring System
After=network.target

[Service]
User=your-username
WorkingDirectory=/path/to/fetal_health_monitoring
Environment="PATH=/path/to/fetal_health_monitoring/venv/bin"
ExecStart=/path/to/fetal_health_monitoring/venv/bin/streamlit run app.py

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl enable fetal-health.service
sudo systemctl start fetal-health.service
sudo systemctl status fetal-health.service
```

---

## 3. Cloud Deployment

### Streamlit Cloud (Easiest Option)

Streamlit Cloud provides free hosting for Streamlit applications and is the simplest deployment option.

#### Step 1: Prepare Your Repository
1. Create a GitHub repository for your project
2. Upload all project files (app.py, requirements.txt, README.md)
3. Ensure the repository is public or you have a Streamlit Cloud subscription for private repos

#### Step 2: Deploy to Streamlit Cloud
1. Visit share.streamlit.io
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch, and main file (app.py)
5. Click "Deploy"

The application will be live at a URL like `your-app-name.streamlit.app` within a few minutes.

### AWS Deployment (EC2)

For more control and institutional requirements, deploy to Amazon Web Services.

#### Step 1: Launch EC2 Instance
1. Log into AWS Console
2. Launch a new EC2 instance (Ubuntu 22.04 LTS recommended)
3. Choose instance type (t2.medium or larger recommended)
4. Configure security group to allow:
   - SSH (port 22) from your IP
   - HTTP (port 80) from anywhere
   - Custom TCP (port 8501) from anywhere

#### Step 2: Connect and Setup
SSH into your instance:
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

Update system and install dependencies:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv nginx -y
```

#### Step 3: Deploy Application
Clone or upload your application files:
```bash
git clone your-repository-url
cd fetal_health_monitoring
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Step 4: Set Up Nginx as Reverse Proxy
Create Nginx configuration at `/etc/nginx/sites-available/fetal-health`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/fetal-health /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### Step 5: Set Up System Service
Create the systemd service as described in the Network Deployment section, then start it.

Your application will now be accessible via your domain name on port 80.

---

## 4. Docker Deployment

Docker containerization ensures consistent deployment across all environments.

### Step 1: Create Dockerfile

Create a file named `Dockerfile` in your project directory:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Create .dockerignore

Create a `.dockerignore` file to exclude unnecessary files:

```
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.git/
.gitignore
README.md
.DS_Store
```

### Step 3: Build Docker Image

```bash
docker build -t fetal-health-monitor .
```

### Step 4: Run Docker Container

```bash
docker run -p 8501:8501 fetal-health-monitor
```

For persistent deployment with auto-restart:

```bash
docker run -d --restart unless-stopped -p 8501:8501 --name fetal-monitor fetal-health-monitor
```

### Docker Compose (Recommended for Production)

Create a `docker-compose.yml` file for easier management:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    restart: unless-stopped
    environment:
      - TZ=America/New_York
    volumes:
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Deploy with Docker Compose:

```bash
docker-compose up -d
```

---

## 5. Security Considerations

### Authentication and Access Control

For production deployment, implement authentication. Streamlit doesn't include built-in authentication, so you have several options:

#### Option 1: Streamlit-Authenticator Library

Install the library:
```bash
pip install streamlit-authenticator
```

Add authentication to your app.py (insert at the beginning of the main function):

```python
import streamlit_authenticator as stauth

# Configure credentials (use hashed passwords in production)
credentials = {
    'usernames': {
        'doctor1': {
            'name': 'Dr. Smith',
            'password': 'hashed_password_here'  # Use bcrypt to hash
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    'fetal_health_monitor',
    'auth_key_123',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status == False:
    st.error('Username/password is incorrect')
    st.stop()
elif authentication_status == None:
    st.warning('Please enter your username and password')
    st.stop()
```

#### Option 2: Nginx Basic Authentication

For simpler requirements, use Nginx basic auth:

```bash
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd doctor1
```

Update your Nginx configuration to include:

```nginx
location / {
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8501;
    # ... rest of proxy settings
}
```

### SSL/TLS Encryption

For production deployment, always use HTTPS to encrypt data in transit.

#### Using Let's Encrypt (Free SSL Certificate)

Install Certbot:
```bash
sudo apt install certbot python3-certbot-nginx
```

Obtain and install certificate:
```bash
sudo certbot --nginx -d your-domain.com
```

Certbot will automatically configure Nginx for HTTPS and set up auto-renewal.

### Data Privacy

Implement the following measures to protect patient data:

1. **Disable Streamlit Telemetry**: Already configured in config.toml
2. **Clear Session Data**: Ensure session state is cleared between patients
3. **No Persistent Storage**: By default, the app doesn't store data; maintain this unless required
4. **Audit Logging**: Add logging for all predictions:

```python
import logging

logging.basicConfig(
    filename='fetal_monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# In your prediction function:
logging.info(f"Prediction made for Patient {patient_id}: {prediction}")
```

---

## 6. Troubleshooting

### Common Issues and Solutions

#### Port Already in Use

**Error:** "Address already in use"

**Solution:** Either stop the other process using port 8501 or specify a different port:

```bash
streamlit run app.py --server.port 8502
```

#### Module Not Found Errors

**Error:** "ModuleNotFoundError: No module named 'streamlit'"

**Solution:** Ensure your virtual environment is activated and dependencies are installed:

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### Permission Denied (Linux/macOS)

**Error:** Permission denied when trying to access port 80 or create files

**Solution:** Either use ports above 1024 (like 8501) or run with elevated privileges (not recommended for port access). For file permissions:

```bash
sudo chown -R $USER:$USER /path/to/project
```

#### Browser Doesn't Open Automatically

**Solution:** Manually navigate to the URL shown in the terminal output, typically `http://localhost:8501`.

#### Application Crashes or Freezes

**Debugging Steps:**
1. Check the terminal output for error messages
2. Verify all input data is within valid ranges
3. Clear browser cache and cookies
4. Restart the Streamlit server
5. Check available system memory (app requires ~500MB)

#### Slow Performance

**Solutions:**
- Use a more powerful instance type for cloud deployments
- Optimize data processing by caching results
- Limit the number of concurrent users
- Consider using a production WSGI server instead of Streamlit's dev server

### Getting Help

If you encounter issues not covered here:

1. Check the terminal output for detailed error messages
2. Review Streamlit documentation at docs.streamlit.io
3. Search for similar issues on Stack Overflow
4. Contact support with detailed error logs and system information

---

## Performance Optimization

For high-traffic deployments:

### Caching

Add caching to expensive operations:

```python
@st.cache_data
def load_model():
    # Load your ML model
    return model

@st.cache_data(ttl=3600)
def process_historical_data(patient_id):
    # Process patient history
    return processed_data
```

### Load Balancing

For very high traffic, deploy multiple instances behind a load balancer:

1. Deploy multiple containers/instances
2. Use Nginx or cloud load balancer to distribute traffic
3. Ensure session persistence if needed

### Monitoring

Set up monitoring to track:
- Application uptime
- Response times
- Error rates
- Resource usage (CPU, memory, disk)

Tools to consider:
- Prometheus + Grafana for metrics
- Sentry for error tracking
- Application logs with ELK stack (Elasticsearch, Logstash, Kibana)

---

## Production Checklist

Before deploying to production:

- [ ] All dependencies listed in requirements.txt
- [ ] Authentication implemented
- [ ] HTTPS/SSL configured
- [ ] Firewall rules properly set
- [ ] Regular backups configured (if storing data)
- [ ] Monitoring and alerting set up
- [ ] Error logging implemented
- [ ] Documentation updated
- [ ] System validated with test data
- [ ] Regulatory compliance verified
- [ ] User training completed
- [ ] Support processes established

---

This deployment guide provides a comprehensive foundation for getting your Fetal Health Monitoring System running in various environments. Remember to always follow your institution's IT policies and regulatory requirements when deploying healthcare applications.
