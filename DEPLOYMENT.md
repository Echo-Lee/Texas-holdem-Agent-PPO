# Deployment Guide - Hugging Face Spaces

Deploy your Texas Hold'em PPO agent to Hugging Face Spaces for free hosting with no size limits.

## Why Hugging Face Spaces?

✅ **No size limits** - PyTorch and large models are fine!
✅ **Free hosting** - No credit card required
✅ **Built-in Gradio support** - Zero configuration needed
✅ **Automatic deployment** - Git push and go live
✅ **Custom domains** - Available for your Space

## Prerequisites

- Hugging Face account (free): https://huggingface.co/join
- Git installed on your machine
- Trained model file (`.pth`)
- Poker card images in `poker_cards/` directory

## Method 1: Automated Deployment (Recommended)

Use the provided script for one-command deployment:

```bash
# Make the script executable
chmod +x deploy_to_hf.sh

# Deploy (replace with your credentials)
./deploy_to_hf.sh YOUR_USERNAME texas-holdem-ppo
```

The script will:
1. Clone your Space repository
2. Copy all necessary files
3. Setup Git LFS for large files
4. Commit and push to Hugging Face

## Method 2: Manual Deployment

### Step 1: Create Your Space

1. Go to https://huggingface.co/new-space
2. Fill in the form:
   - **Space name**: `texas-holdem-ppo` (or your choice)
   - **License**: MIT
   - **Select SDK**: Choose **"Gradio"**
   - **Gradio version**: Use default (or 5.16.0)
   - **Space hardware**: CPU (free tier)
   - **Visibility**: Public or Private
3. Click **"Create Space"**

### Step 2: Clone Your Space

```bash
# Clone the empty Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME
```

### Step 3: Copy Project Files

```bash
# Copy the Space README (with YAML front matter)
cp ../RL\ Project/ HF_README.md README.md

# Copy application code
cp ../RL\ Project/app.py .
cp -r ../RL\ Project/core .
cp ../RL\ Project/running_config.yaml .

# Copy trained model
mkdir -p models
cp ../RL\ Project/models/SP-U20_w-OppM_lr0.0003_final.pth models/

# Copy card images
cp -r ../RL\ Project/poker_cards .

# Copy dependencies
cp ../RL\ Project/requirements.txt .
```

### Step 4: Setup Git LFS (for Large Files)

Git LFS is required for files larger than 10MB (models, images):

```bash
# Install Git LFS (if not already installed)
git lfs install

# Copy LFS configuration
cp ../RL\ Project/.gitattributes .

# Verify LFS is tracking large files
cat .gitattributes
# Should show: *.pth filter=lfs diff=lfs merge=lfs -text
```

### Step 5: Commit and Push

```bash
# Stage all files
git add .

# Commit with a descriptive message
git commit -m "Deploy Texas Hold'em PPO Agent with dark casino theme"

# Push to Hugging Face
git push
```

### Step 6: Wait for Build

Your Space will automatically:
1. Install dependencies from `requirements.txt`
2. Download large files via Git LFS
3. Launch the Gradio app
4. Be live at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

**Build time**: Usually 2-5 minutes

## Troubleshooting

### Build Fails - Missing Dependencies

Check `requirements.txt` is correct. Should contain:
```txt
torch==2.10.0
numpy==2.4.2
pettingzoo==1.25.0
gymnasium==1.2.3
gradio==5.16.0
PyYAML==6.0.3
pygame==2.6.1
pillow==12.1.1
```

### Model Not Loading

1. Verify model path in `app.py`:
   ```python
   MODEL_PATH = os.getenv("MODEL_PATH", "models/SP-U20_w-OppM_lr0.0003_final.pth")
   ```

2. Check model file exists in `models/` directory

3. Ensure Git LFS is tracking `.pth` files:
   ```bash
   git lfs ls-files
   ```

### Card Images Not Showing

1. Verify `poker_cards/` directory contains all card images
2. Check image format is `.png`
3. Naming convention: `AS.png`, `2H.png`, `BACK.png`, etc.

### "Space is building..." Takes Too Long

- Check build logs in your Space's "Logs" tab
- Large models (~700MB) may take 5-10 minutes on first build
- Subsequent builds are faster (cached)

## Updating Your Deployment

To update your deployed Space:

```bash
# Navigate to your Space directory
cd YOUR_SPACE_NAME

# Make changes to files
# Then commit and push
git add .
git commit -m "Update: description of changes"
git push
```

The Space will automatically rebuild with your changes.

## Configuration Options

### Custom Model Path

Set environment variable in Space settings:
1. Go to your Space → Settings → Variables
2. Add: `MODEL_PATH` = `models/your_model.pth`

### Change Starting Chips

Edit `app.py`:
```python
DEFAULT_INITIAL_STACK = 100  # Change this value
```

### Adjust UI Theme

Modify the color scheme in `app.py`:
```python
casino_theme = gr.themes.Base(
    primary_hue="emerald",     # Change poker table color
    secondary_hue="amber",     # Change accent color
    neutral_hue="slate",       # Change neutral colors
)
```

## Testing Locally Before Deployment

Always test locally first:

```bash
# Navigate to project directory
cd "d:\RL Project"

# Run the app
python app.py

# Visit in browser
# http://localhost:7860
```

Verify:
- ✅ Model loads without errors
- ✅ Cards display correctly
- ✅ Actions work properly
- ✅ UI looks good

## Space Settings

### Make Space Public/Private

1. Go to your Space → Settings
2. Change "Visibility" setting
3. Save changes

### Add Custom Domain

Hugging Face Spaces Pro tier supports custom domains.

### Hardware Upgrades

Free tier uses CPU. For faster inference:
1. Go to Space → Settings → Hardware
2. Upgrade to GPU (paid)

## Resources

- **HF Spaces Documentation**: https://huggingface.co/docs/hub/spaces
- **Git LFS Guide**: https://git-lfs.github.com/
- **Gradio Documentation**: https://gradio.app/docs/
- **Your Space URL**: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE`

## Support

If you encounter issues:
1. Check build logs in Space → Logs tab
2. Verify all files are committed
3. Test locally with `python app.py`
4. Open an issue on GitHub: https://github.com/Echo-Lee/Leduc-holdem-Agent-PPO/issues

---

**Happy deploying! Your poker agent will be live in minutes!** 🎰🚀
