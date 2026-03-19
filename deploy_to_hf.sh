#!/bin/bash

# Deploy to Hugging Face Space
# Usage: bash deploy_to_hf.sh

set -e

SPACE_URL="https://huggingface.co/spaces/ChenyuEcho/texas-holdem-ppo"
TEMP_DIR="hf_space_temp"

echo "🚀 Deploying to Hugging Face Space..."

# 1. Clone the HF Space repository
echo "📥 Cloning Space repository..."
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi
git clone "$SPACE_URL" "$TEMP_DIR"
cd "$TEMP_DIR"

# 2. Setup Git LFS for model files
echo "🔧 Setting up Git LFS..."
git lfs install
git lfs track "*.pt"
git add .gitattributes

# 3. Copy necessary files
echo "📁 Copying files..."
cp ../app.py .
cp ../requirements.txt .
cp ../running_config.yaml .
cp ../running_config_mlp.yaml .

# Copy README for HF Space
cp ../HF_SPACE_README.md ./README.md

# Copy core directory
cp -r ../core .

# Copy poker cards
cp -r ../poker_cards .

# Copy models
mkdir -p models
cp ../models/*.pt models/ 2>/dev/null || echo "⚠️  No model files found"

# 4. Create .gitignore if needed
cat > .gitignore << 'INNER_EOF'
__pycache__/
*.pyc
.venv/
.idea/
.vscode/
*.log
results/
results_mlp/
scripts/
INNER_EOF

# 5. Commit and push
echo "💾 Committing changes..."
git add .
git commit -m "Update: Add model selection, pot/to_call display, fixed card order" || echo "No changes to commit"

echo "📤 Pushing to Hugging Face..."
git push

cd ..
echo "✅ Deployment complete! Visit: $SPACE_URL"
