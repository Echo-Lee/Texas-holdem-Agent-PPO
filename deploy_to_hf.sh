#!/bin/bash
# Quick deployment script for Hugging Face Spaces

set -e

echo "🎰 Texas Hold'em PPO Agent - Hugging Face Spaces Deployment"
echo "=========================================================="
echo ""

# Check if HF username and space name are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./deploy_to_hf.sh YOUR_USERNAME YOUR_SPACE_NAME"
    echo "Example: ./deploy_to_hf.sh myusername texas-holdem-ppo"
    exit 1
fi

HF_USERNAME=$1
SPACE_NAME=$2
SPACE_URL="https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

echo "📦 Preparing deployment to:"
echo "   $SPACE_URL"
echo ""

# Check if Space directory exists
if [ -d ".hf-deploy/$SPACE_NAME" ]; then
    echo "⚠️  Directory .hf-deploy/$SPACE_NAME already exists."
    read -p "Delete and re-clone? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ".hf-deploy/$SPACE_NAME"
    else
        echo "❌ Aborted."
        exit 1
    fi
fi

# Create deployment directory
mkdir -p .hf-deploy

# Clone the Space
echo "📥 Cloning Space repository..."
git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" ".hf-deploy/$SPACE_NAME"
cd ".hf-deploy/$SPACE_NAME"

# Copy README
echo "📝 Copying README..."
cp "../../HF_README.md" README.md

# Copy application files
echo "📂 Copying application files..."
cp "../../app.py" .
cp -r "../../core" .
cp "../../running_config.yaml" .
cp "../../requirements.txt" .

# Copy model (if exists)
if [ -f "../../models/SP-U20_w-OppM_lr0.0003_final.pth" ]; then
    echo "🤖 Copying trained model..."
    mkdir -p models
    cp "../../models/SP-U20_w-OppM_lr0.0003_final.pth" models/
else
    echo "⚠️  Warning: Model not found at models/SP-U20_w-OppM_lr0.0003_final.pth"
    echo "   Please add your trained model before deployment!"
fi

# Copy card images
if [ -d "../../poker_cards" ]; then
    echo "🃏 Copying poker card images..."
    cp -r "../../poker_cards" .
else
    echo "⚠️  Warning: poker_cards/ directory not found"
fi

# Setup Git LFS
echo "📦 Setting up Git LFS..."
git lfs install
cp "../../.gitattributes" .

# Commit and push
echo "🚀 Committing and pushing to Hugging Face..."
git add .
git commit -m "Deploy Texas Hold'em PPO Agent with dark casino theme"
git push

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your Space will be live at:"
echo "   $SPACE_URL"
echo ""
echo "⏳ Build typically takes 2-5 minutes..."
echo "📊 Check build status at: $SPACE_URL"
echo ""
