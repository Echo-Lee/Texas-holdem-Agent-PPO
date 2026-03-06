# Deployment Guide for Vercel

## ⚠️ Important Constraints

Vercel has strict deployment limits:
- **50MB** deployment size limit (zipped)
- **250MB** unzipped size limit
- **Serverless functions** are stateless

**PyTorch is ~700MB**, which exceeds Vercel's limits. You have several options:

## Option 1: Use Hugging Face Spaces (Recommended)

Hugging Face Spaces is better suited for ML apps with large dependencies.

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Create a new Space
# Then copy app.py, core/, running_config.yaml, and your model to the Space
```

Create a `README.md` in your Space:
```yaml
---
title: Texas Holdem PPO Agent
emoji: 🃏
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
---
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
