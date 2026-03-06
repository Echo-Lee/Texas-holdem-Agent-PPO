# Deployment Guide

## 🏆 Hugging Face Spaces (Recommended & Easiest)

Hugging Face Spaces is purpose-built for ML apps with large dependencies. This is now the primary deployment method for this project.

### Why Hugging Face Spaces?
- ✅ **No size limits** - PyTorch is fine!
- ✅ **Free hosting** - No credit card required
- ✅ **Built-in Gradio support** - Zero configuration
- ✅ **Automatic deployment** - Git push and go live
- ✅ **Custom domains** available

## Quick Deployment to Hugging Face Spaces

### Step 1: Create Your Space

1. Go to https://huggingface.co/new-space
2. Choose a name (e.g., "texas-holdem-ppo-agent")
3. Select **"Gradio"** as the SDK
4. Choose **"Public"** (free) or **"Private"**
5. Click **"Create Space"**

### Step 2: Prepare Your Files

Make sure you have:
- ✅ `app.py` (dark casino themed UI)
- ✅ `core/` directory (all Python modules)
- ✅ `running_config.yaml` (configuration)
- ✅ `models/SP-U20_w-OppM_lr0.0003_final.pth` (your trained model)
- ✅ `poker_cards/` directory (card images)
- ✅ `requirements.txt` (dependencies)
- ✅ `HF_README.md` (Space description)

### Step 3: Deploy via Git

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy README for Space
cp ../HF_README.md README.md

# Copy your files
cp -r ../app.py ../core ../running_config.yaml ../models ../poker_cards ../requirements.txt .

# Commit and push
git add .
git commit -m "Initial deployment with dark casino theme"
git push
```

### Step 4: Wait for Build

Your Space will automatically:
1. Install dependencies
2. Load your model
3. Launch the Gradio interface
4. Be live at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

Usually takes 2-5 minutes!

### Step 5: (Optional) Use Git LFS for Large Files

If your model is very large (>10MB), use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.png"

# Add and commit
git add .gitattributes
git commit -m "Track large files with LFS"
git push
```

## Option 2: Convert Model to ONNX (Smaller)

ONNX Runtime is much smaller than PyTorch:

```python
# Export your model to ONNX format
import torch
from core.networks.policy_value_network import PolicyNet

model = PolicyNet(input_dim=76, output_dim=4, hidden_layers=[256, 256], use_layer_norm=True)
model.load_state_dict(torch.load("models/SP-U20_w-OppM_lr0.0003_final.pth"))
model.eval()

dummy_input = torch.randn(1, 76)
torch.onnx.export(model, dummy_input, "models/policy.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
```

Then modify `app.py` to use ONNX Runtime instead of PyTorch.

Update `requirements-vercel.txt`:
```
gradio==5.16.0
numpy==2.4.2
pettingzoo==1.25.0
PyYAML==6.0.3
onnxruntime==1.20.0
```

## Option 3: Deploy to Railway/Render

These platforms have more generous limits:

### Railway
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and init
railway login
railway init

# Deploy
railway up
```

### Render
1. Connect your GitHub repo
2. Create a new Web Service
3. Build command: `pip install -r requirements.txt`
4. Start command: `python app.py`

## Option 4: Try Vercel with CPU-only PyTorch

Update `requirements-vercel.txt`:
```
gradio==5.16.0
numpy==2.4.2
pettingzoo==1.25.0
PyYAML==6.0.3
torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

**Note**: This may still exceed limits. Monitor deployment size.

## Vercel Deployment Steps (if within size limits)

1. **Prepare poker card images**:
   ```bash
   mkdir -p public/poker_cards
   # Copy your card images to public/poker_cards/
   ```

2. **Copy your trained model**:
   ```bash
   # Make sure your model is at: models/SP-U20_w-OppM_lr0.0003_final.pth
   ```

3. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

4. **Deploy**:
   ```bash
   vercel login
   vercel
   ```

5. **Set environment variables** (if needed):
   ```bash
   vercel env add MODEL_PATH
   # Enter: models/SP-U20_w-OppM_lr0.0003_final.pth
   ```

## Testing Locally

```bash
# Test the app locally
python app.py

# Or with Vercel dev server
vercel dev
```

## File Structure for Deployment

```
d:\RL Project/
├── api/
│   └── index.py          # Vercel serverless function entry
├── core/                 # Your core package
├── models/               # Trained models
├── public/
│   └── poker_cards/      # Card images
├── app.py                # Main Gradio app
├── running_config.yaml   # Config file
├── requirements-vercel.txt
├── vercel.json
└── .vercelignore
```

## Recommended: Hugging Face Spaces

Given the size constraints, **Hugging Face Spaces** is the most straightforward option:

1. Create a Space at https://huggingface.co/new-space
2. Upload these files:
   - `app.py`
   - `core/` directory
   - `running_config.yaml`
   - `models/SP-U20_w-OppM_lr0.0003_final.pth`
   - `public/poker_cards/` directory
   - Create `requirements.txt` with full dependencies

Your Space will be live at `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`
