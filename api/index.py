# Vercel serverless function entry point
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Gradio app
from app import demo

# Vercel requires a handler function
app = demo.queue()
