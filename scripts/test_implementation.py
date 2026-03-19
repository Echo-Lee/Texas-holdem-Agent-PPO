"""
Quick test to verify the implementation works
"""
import torch
import yaml
from pathlib import Path

from core.networks import PolicyNet, ValueNet, FlexibleNet
from core.main import (
    load_config,
    build_agent_network_kwargs,
    create_ppo_agent,
    detect_env_dims,
)


def test_config_loading():
    """Test that both configs load correctly"""
    print("Testing config loading...")

    cnn_cfg = load_config("running_config.yaml")
    assert cnn_cfg["system"]["model_type"] == "cnn"
    print("  ✓ CNN config loaded")

    mlp_cfg = load_config("running_config_mlp.yaml")
    assert mlp_cfg["system"]["model_type"] == "mlp"
    print("  ✓ MLP config loaded")

    print()


def test_network_creation():
    """Test that both network types can be created"""
    print("Testing network creation...")

    input_dim = 76
    action_dim = 4

    # CNN
    cnn_policy = PolicyNet(
        input_dim=input_dim,
        output_dim=action_dim,
        hidden_layers=[256, 256, 128],
        use_layer_norm=True,
        card_encoder_channels=[16, 32],
        card_embedding_dim=128,
    )
    assert cnn_policy is not None
    print("  ✓ CNN PolicyNet created")

    cnn_value = ValueNet(
        input_dim=input_dim,
        hidden_layers=[256, 256, 128],
        use_layer_norm=True,
        card_encoder_channels=[16, 32],
        card_embedding_dim=128,
    )
    assert cnn_value is not None
    print("  ✓ CNN ValueNet created")

    # MLP
    mlp_policy = FlexibleNet(
        input_dim=input_dim,
        output_dim=action_dim,
        hidden_layers=[512, 400, 256],
        use_layer_norm=True,
    )
    assert mlp_policy is not None
    print("  ✓ MLP PolicyNet created")

    mlp_value = FlexibleNet(
        input_dim=input_dim,
        output_dim=1,
        hidden_layers=[512, 400, 256],
        use_layer_norm=True,
    )
    assert mlp_value is not None
    print("  ✓ MLP ValueNet created")

    print()


def test_forward_pass():
    """Test that networks can process inputs"""
    print("Testing forward pass...")

    batch_size = 32
    input_dim = 76
    action_dim = 4

    # Create dummy input
    x = torch.randn(batch_size, input_dim)

    # CNN
    cnn_policy = PolicyNet(
        input_dim=input_dim,
        output_dim=action_dim,
        hidden_layers=[256, 256, 128],
        use_layer_norm=True,
        card_encoder_channels=[16, 32],
        card_embedding_dim=128,
    )

    cnn_output = cnn_policy(x)
    assert cnn_output.shape == (batch_size, action_dim)
    print("  ✓ CNN forward pass works")

    # MLP
    mlp_policy = FlexibleNet(
        input_dim=input_dim,
        output_dim=action_dim,
        hidden_layers=[512, 400, 256],
        use_layer_norm=True,
    )

    mlp_output = mlp_policy(x)
    assert mlp_output.shape == (batch_size, action_dim)
    print("  ✓ MLP forward pass works")

    print()


def test_agent_creation():
    """Test that agents can be created from configs"""
    print("Testing agent creation...")

    device = "cpu"
    raw_obs_dim, action_dim = detect_env_dims()
    agent_obs_dim = raw_obs_dim + action_dim

    # CNN Agent
    cnn_cfg = load_config("running_config.yaml")
    cnn_agent = create_ppo_agent(cnn_cfg, agent_obs_dim, action_dim, device)
    assert cnn_agent is not None
    print("  ✓ CNN agent created from config")

    # MLP Agent
    mlp_cfg = load_config("running_config_mlp.yaml")
    mlp_agent = create_ppo_agent(mlp_cfg, agent_obs_dim, action_dim, device)
    assert mlp_agent is not None
    print("  ✓ MLP agent created from config")

    print()


def test_parameter_counts():
    """Test that parameter counts are comparable"""
    print("Testing parameter counts...")

    input_dim = 76
    action_dim = 4

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    # CNN
    cnn_policy = PolicyNet(
        input_dim=input_dim,
        output_dim=action_dim,
        hidden_layers=[256, 256, 128],
        use_layer_norm=True,
        card_encoder_channels=[16, 32],
        card_embedding_dim=128,
    )
    cnn_params = count_params(cnn_policy)

    # MLP
    mlp_policy = FlexibleNet(
        input_dim=input_dim,
        output_dim=action_dim,
        hidden_layers=[512, 400, 256],
        use_layer_norm=True,
    )
    mlp_params = count_params(mlp_policy)

    diff_pct = abs(cnn_params - mlp_params) / cnn_params * 100

    print(f"  CNN PolicyNet: {cnn_params:,} params")
    print(f"  MLP PolicyNet: {mlp_params:,} params")
    print(f"  Difference: {diff_pct:.1f}%")

    assert diff_pct < 5, f"Parameter difference too large: {diff_pct:.1f}%"
    print("  ✓ Parameter counts are comparable")

    print()


def main():
    print("=" * 60)
    print("Implementation Test Suite")
    print("=" * 60)
    print()

    try:
        test_config_loading()
        test_network_creation()
        test_forward_pass()
        test_agent_creation()
        test_parameter_counts()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("You can now:")
        print("  1. Train CNN agent: python core/main.py --config running_config.yaml")
        print("  2. Train MLP agent: python core/main.py --config running_config_mlp.yaml")
        print("  3. Compare agents: python agent_battle.py --agent1 <path1> --agent2 <path2>")
        print()

    except Exception as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
