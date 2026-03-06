#!/usr/bin/env python
"""
Helper script to prepare the project for deployment.
"""
import os
import shutil
from pathlib import Path

def setup_deployment():
    """Set up deployment structure."""
    print("🎯 Setting up deployment structure...\n")

    # Create public directory for card images
    public_dir = Path("public/poker_cards")
    public_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {public_dir}")

    # Check if poker_cards exists in the current directory
    source_dir = Path("poker_cards")
    if source_dir.exists():
        print(f"\n📋 Found poker_cards directory. Copying to public/...")
        for item in source_dir.glob("*.png"):
            dest = public_dir / item.name
            shutil.copy2(item, dest)
            print(f"  ✓ Copied: {item.name}")
        print(f"✓ Copied {len(list(public_dir.glob('*.png')))} card images")
    else:
        print(f"\n⚠️  Warning: poker_cards/ directory not found")
        print(f"   Please copy your card images to: {public_dir}")
        print(f"   Expected format: AS.png, 2H.png, BACK.png, etc.")

    # Check for trained model
    model_path = Path("models/SP-U20_w-OppM_lr0.0003_final.pth")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Found trained model: {model_path}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"\n⚠️  Warning: Model not found at {model_path}")
        print(f"   Update MODEL_PATH in app.py to point to your model")

    # Check requirements
    print(f"\n📦 Checking dependencies...")
    req_file = Path("requirements.txt")
    req_vercel = Path("requirements-vercel.txt")

    if req_file.exists():
        print(f"  ✓ requirements.txt found (for local development)")
    if req_vercel.exists():
        print(f"  ✓ requirements-vercel.txt found (for deployment)")

    # Summary
    print(f"\n{'='*60}")
    print("📊 Deployment Checklist")
    print(f"{'='*60}")

    checklist = [
        ("Card images in public/poker_cards/", public_dir.exists() and len(list(public_dir.glob("*.png"))) > 0),
        ("Trained model available", model_path.exists()),
        ("app.py created", Path("app.py").exists()),
        ("vercel.json created", Path("vercel.json").exists()),
        ("Deployment guide available", Path("DEPLOYMENT.md").exists()),
    ]

    for item, status in checklist:
        symbol = "✓" if status else "❌"
        print(f"  {symbol} {item}")

    print(f"\n{'='*60}")
    print("🚀 Next Steps")
    print(f"{'='*60}")

    if all(status for _, status in checklist):
        print("""
✓ All checks passed! You're ready to deploy.

Recommended deployment options:

1. Hugging Face Spaces (Easiest):
   - Visit https://huggingface.co/new-space
   - Upload app.py, core/, running_config.yaml, models/, public/
   - Your app goes live automatically

2. Convert to ONNX (Smaller size):
   python convert_to_onnx.py --model models/SP-U20_w-OppM_lr0.0003_final.pth --test

3. Test locally first:
   python app.py

See DEPLOYMENT.md for detailed instructions.
        """)
    else:
        print("""
⚠️  Some items need attention:

- If card images are missing, copy them to public/poker_cards/
- If model is missing, train one or update MODEL_PATH in app.py
- See DEPLOYMENT.md for detailed setup instructions
        """)

    print(f"{'='*60}\n")


if __name__ == "__main__":
    setup_deployment()
