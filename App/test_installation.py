"""
Test script to verify the Fetal Health Monitoring System installation
Run this script to check if all dependencies are installed correctly.

Usage:
    python test_installation.py
"""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    print("-" * 50)
    
    packages = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'plotly': 'Plotly',
    }
    
    all_passed = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            version = __import__(package).__version__
            print(f"✓ {name:20} version {version}")
        except ImportError as e:
            print(f"✗ {name:20} NOT FOUND - {str(e)}")
            all_passed = False
    
    print("-" * 50)
    return all_passed


def test_python_version():
    """Test if Python version is compatible."""
    print("\nTesting Python version...")
    print("-" * 50)
    
    major = sys.version_info.major
    minor = sys.version_info.minor
    
    print(f"Python version: {major}.{minor}.{sys.version_info.micro}")
    
    if major >= 3 and minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python version is too old (requires 3.8+)")
        return False


def test_app_file():
    """Test if app.py exists and can be loaded."""
    print("\nTesting app.py...")
    print("-" * 50)
    
    try:
        with open('app.py', 'r') as f:
            content = f.read()
            if 'def main()' in content:
                print("✓ app.py found and contains main function")
                return True
            else:
                print("✗ app.py missing main function")
                return False
    except FileNotFoundError:
        print("✗ app.py not found")
        return False


def test_requirements_file():
    """Test if requirements.txt exists."""
    print("\nTesting requirements.txt...")
    print("-" * 50)
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            print("✓ requirements.txt found")
            print("\nPackages in requirements.txt:")
            for line in requirements.strip().split('\n'):
                if line and not line.startswith('#'):
                    print(f"  - {line}")
            return True
    except FileNotFoundError:
        print("✗ requirements.txt not found")
        return False


def test_config_file():
    """Test if .streamlit/config.toml exists."""
    print("\nTesting configuration...")
    print("-" * 50)
    
    try:
        with open('.streamlit/config.toml', 'r') as f:
            print("✓ .streamlit/config.toml found")
            return True
    except FileNotFoundError:
        print("⚠ .streamlit/config.toml not found (optional)")
        return True  # This is optional, so return True


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("FETAL HEALTH MONITORING SYSTEM - INSTALLATION TEST")
    print("=" * 50)
    
    tests = [
        test_python_version(),
        test_imports(),
        test_app_file(),
        test_requirements_file(),
        test_config_file()
    ]
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    if all(tests):
        print("\n✅ All tests passed!")
        print("\nYou can now run the application:")
        print("    streamlit run app.py")
    else:
        print("\n❌ Some tests failed!")
        print("\nPlease fix the issues above before running the application.")
        print("\nCommon fixes:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Ensure you're in the correct directory")
        print("  3. Check Python version: python --version")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
