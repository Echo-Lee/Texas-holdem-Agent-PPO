#!/usr/bin/env python
"""
Convert PyTorch model to ONNX format for smaller deployment size.
ONNX Runtime (~10MB) vs PyTorch (~700MB)
"""
import torch
import yaml
from pathlib import Path
from core.networks.policy_value_network import PolicyNet

def convert_model_to_onnx(
    model_path: str,
    config_path: str = "running_config.yaml",
    output_path: str = None
):
    """Convert PyTorch policy model to ONNX format."""

    if output_path is None:
        output_path = Path(model_path).with_suffix('.onnx')

    print(f"Loading config from {config_path}...")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    agent_cfg = cfg.get('model', {}).get('agent', {
        'hidden_layers': [256, 256],
        'use_layer_norm': True
    })

    # Texas Hold'em dimensions
    obs_dim = 72  # raw observation
    obs_dim += 4   # opponent action encoding
    action_dim = 4

    print(f"Creating PolicyNet with input_dim={obs_dim}, output_dim={action_dim}")
    print(f"Hidden layers: {agent_cfg['hidden_layers']}")
    print(f"Layer norm: {agent_cfg['use_layer_norm']}")

    # Create model architecture
    model = PolicyNet(
        input_dim=obs_dim,
        output_dim=action_dim,
        hidden_layers=agent_cfg['hidden_layers'],
        use_layer_norm=agent_cfg['use_layer_norm']
    )

    # Load trained weights
    print(f"\nLoading weights from {model_path}...")
    state = torch.load(model_path, map_location='cpu')

    # Handle different save formats
    if isinstance(state, dict):
        if "policy_state_dict" in state:
            state = state["policy_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"Model loaded successfully!")

    # Create dummy input for export
    dummy_input = torch.randn(1, obs_dim)

    # Export to ONNX
    print(f"\nExporting to ONNX format: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['observation'],
        output_names=['logits'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=17,
        do_constant_folding=True
    )

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"✓ ONNX model exported successfully!")
    print(f"✓ Model verified and saved to: {output_path}")

    # Compare file sizes
    pt_size = Path(model_path).stat().st_size / (1024 * 1024)
    onnx_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  PyTorch: {pt_size:.2f} MB")
    print(f"  ONNX:    {onnx_size:.2f} MB")

    return output_path


def test_onnx_inference(onnx_path: str):
    """Test ONNX model inference."""
    import onnxruntime as ort
    import numpy as np

    print(f"\n{'='*60}")
    print("Testing ONNX model inference...")
    print(f"{'='*60}")

    # Create session
    session = ort.InferenceSession(onnx_path)

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")

    # Create test input
    test_input = np.random.randn(1, 76).astype(np.float32)

    # Run inference
    result = session.run([output_name], {input_name: test_input})
    logits = result[0]

    print(f"\nTest input shape: {test_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output logits: {logits}")

    # Apply softmax to get probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    print(f"Action probabilities: {probs}")
    print(f"Predicted action: {np.argmax(probs)}")

    print(f"\n✓ ONNX inference test passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="models/SP-U20_w-OppM_lr0.0003_final.pth",
        help="Path to PyTorch model file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="running_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ONNX model (default: same as model with .onnx extension)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the ONNX model after conversion"
    )

    args = parser.parse_args()

    try:
        onnx_path = convert_model_to_onnx(
            model_path=args.model,
            config_path=args.config,
            output_path=args.output
        )

        if args.test:
            test_onnx_inference(onnx_path)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
