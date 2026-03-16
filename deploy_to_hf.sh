#!/bin/bash
# Quick deployment script for Hugging Face Spaces

set -e

echo "Texas Hold'em PPO Agent - Hugging Face Spaces Deployment"
echo "======================================================="
echo ""

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./deploy_to_hf.sh kailiu0712 texas-holdem-cnnppo"
    echo "Example: ./deploy_to_hf.sh kailiu0712 texas-holdem-cnnppo"
    exit 1
fi

HF_USERNAME=$1
SPACE_NAME=$2
SPACE_URL="https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

echo "Preparing deployment to:"
echo "  $SPACE_URL"
echo ""

if [ -d ".hf-deploy/$SPACE_NAME" ]; then
    echo "Directory .hf-deploy/$SPACE_NAME already exists."
    read -p "Delete and re-clone? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ".hf-deploy/$SPACE_NAME"
    else
        echo "Aborted."
        exit 1
    fi
fi

mkdir -p .hf-deploy

echo "Cloning Space repository..."
git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" ".hf-deploy/$SPACE_NAME"
cd ".hf-deploy/$SPACE_NAME"

echo "Copying README..."
cp "../../HF_README.md" README.md

echo "Copying application files..."
cp "../../app.py" .
cp -r "../../core" .
cp "../../running_config.yaml" .
cp "../../requirements.txt" .

if [ -f "../../models/stage3_final_policy.pt" ]; then
    echo "Copying stage3 model..."
    mkdir -p models
    cp "../../models/stage3_final_policy.pt" models/
else
    echo "Warning: Model not found at models/stage3_final_policy.pt"
    echo "Please add the trained stage3 model before deployment."
fi

if [ -d "../../poker_cards" ]; then
    echo "Copying poker card images..."
    cp -r "../../poker_cards" .
else
    echo "Warning: poker_cards/ directory not found"
fi

echo "Setting up Git LFS..."
git lfs install
cp "../../.gitattributes" .

echo "Committing and pushing to Hugging Face..."
git add .
git commit -m "Deploy stage3 Texas Hold'em PPO Agent"
git push

echo ""
echo "Deployment complete."
echo "Your Space will be live at:"
echo "  $SPACE_URL"
echo ""
echo "Build typically takes a few minutes."
echo "Check build status at: $SPACE_URL"
echo ""
