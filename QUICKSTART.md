# 🚀 Quick Start Guide

Get Vexa Polish LLM up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB free disk space
- Internet connection (for Wikipedia download)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vexa-polish-llm.git
cd vexa-polish-llm

# Install dependencies
pip install -r requirements.txt
```

## Option 1: Full Pipeline (Recommended for First Time)

Run everything with one command:

```bash
python main.py all --articles 50
```

This will:
1. ✅ Download 50 Wikipedia articles (~2 minutes)
2. ✅ Prepare training data (~30 seconds)
3. ✅ Train the model (~10 minutes)
4. ✅ Launch chat interface

**Note**: Training with 50 articles is quick but produces a basic model. For better results, use 200+ articles.

## Option 2: Step-by-Step

### Step 1: Download Data

```bash
python main.py download --articles 100
```

Articles are saved to `data/raw/`.

### Step 2: Prepare Training Data

```bash
python main.py prepare
```

This creates:
- `data/vocab.json` - Character vocabulary
- `data/training_data.bin` - Compressed training sequences

### Step 3: Train Model

```bash
python main.py train
```

Training progress is displayed in real-time. Press `Ctrl+C` to stop (model is auto-saved).

### Step 4: Chat with Model

```bash
python main.py chat
```

## Quick Test (Minimal Setup)

For a quick test with minimal data:

```bash
# Download only 20 articles (fast!)
python main.py download --articles 20

# Prepare and train
python main.py prepare
python main.py train

# Chat
python main.py chat
```

**Expected time**: ~5 minutes total

## Chat Commands

Once in chat mode:

```
You: Hello!
Vexa: [response]

You: /stats          # Show statistics
You: /clear          # Clear conversation history
You: /feedback 0.8   # Rate last response (0-1)
You: /save           # Save conversation
You: /quit           # Exit
```

## Configuration

Edit `config/hyperparams.yaml` to customize:

```yaml
# Quick training (faster, less accurate)
num_epochs: 100
num_ants: 20

# Quality training (slower, more accurate)
num_epochs: 1000
num_ants: 50
```

## Troubleshooting

### "No module named 'src'"

```bash
# Make sure you're in the project directory
cd vexa-polish-llm
python main.py chat
```

### "No training data"

```bash
# Run preparation first
python main.py prepare
```

### Model generates nonsense

```bash
# Train longer or with more data
python main.py download --articles 200
python main.py prepare
python main.py train
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [INSTALL.md](INSTALL.md) for advanced installation
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Need Help?

Open an issue on GitHub with:
- Your Python version (`python --version`)
- Error message (if any)
- Steps you followed

---

**Happy chatting with Vexa!** 🐜🇵🇱
