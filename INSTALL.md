# Installation Guide

Detailed installation instructions for Vexa Polish LLM.

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB
- **Disk Space**: 2GB free
- **Internet**: Required for Wikipedia download

### Recommended Requirements
- **RAM**: 8GB or more
- **CPU**: Multi-core processor
- **Disk Space**: 5GB free (for larger datasets)

## Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/vexa-polish-llm.git
cd vexa-polish-llm

# 2. Create virtual environment (recommended)
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import numpy, yaml, requests, tqdm; print('✓ All dependencies installed')"
```

### Method 2: Development Installation

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/yourusername/vexa-polish-llm.git
cd vexa-polish-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install in editable mode
pip install -e .

# Run tests
python -m pytest tests/
```

### Method 3: Docker (Coming Soon)

```bash
# Pull image
docker pull vexallm/vexa-polish-llm:latest

# Run container
docker run -it vexallm/vexa-polish-llm:latest
```

## Platform-Specific Instructions

### Windows

```powershell
# Install Python from python.org
# Open PowerShell or Command Prompt

# Clone repository
git clone https://github.com/yourusername/vexa-polish-llm.git
cd vexa-polish-llm

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: If you encounter SSL errors, try:
```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### macOS

```bash
# Install Python via Homebrew (if not installed)
brew install python@3.9

# Clone repository
git clone https://github.com/yourusername/vexa-polish-llm.git
cd vexa-polish-llm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Clone repository
git clone https://github.com/yourusername/vexa-polish-llm.git
cd vexa-polish-llm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dependency Details

### Core Dependencies

```
numpy>=1.21.0          # Numerical computations
pyyaml>=5.4.1          # Configuration files
requests>=2.26.0       # Wikipedia API
tqdm>=4.62.0           # Progress bars
```

### Development Dependencies

```
pytest>=7.0.0          # Testing framework
black>=22.0.0          # Code formatting
flake8>=4.0.0          # Linting
mypy>=0.950            # Type checking
```

## Verification

After installation, verify everything works:

```bash
# Test imports
python -c "from src.core import AntGraph, VexaEngine; print('✓ Core modules OK')"
python -c "from src.utils import VexaTokenizer; print('✓ Utils modules OK')"
python -c "from src.integration import VexaLLM; print('✓ Integration modules OK')"

# Test configuration
python -c "import yaml; yaml.safe_load(open('config/hyperparams.yaml')); print('✓ Config OK')"

# Quick functionality test
python main.py --help
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution**: Make sure you're in the project root directory:
```bash
cd vexa-polish-llm
python main.py chat
```

### Issue: "pip: command not found"

**Solution**: Install pip:
```bash
# macOS/Linux
python3 -m ensurepip --upgrade

# Windows
python -m ensurepip --upgrade
```

### Issue: "Permission denied" on Linux/macOS

**Solution**: Don't use sudo with pip in virtual environment:
```bash
# Activate venv first
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: SSL Certificate Error

**Solution**:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Issue: Out of Memory During Training

**Solution**: Reduce parameters in `config/hyperparams.yaml`:
```yaml
num_ants: 20          # Reduce from 50
batch_size: 5         # Reduce from 10
sequence_length: 50   # Reduce from 100
```

## Updating

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Verify
python -c "import src; print('✓ Update successful')"
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove project directory
cd ..
rm -rf vexa-polish-llm

# Or on Windows:
# rmdir /s vexa-polish-llm
```

## Next Steps

After successful installation:

1. Read [QUICKSTART.md](QUICKSTART.md) for quick start guide
2. Read [README.md](README.md) for full documentation
3. Run `python main.py all --articles 50` for first test

## Support

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub Issues](https://github.com/yourusername/vexa-polish-llm/issues)
3. Open a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

---

**Installation complete!** Ready to train your ACO-based LLM! 🐜
