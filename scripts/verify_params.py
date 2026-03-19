"""
Verify parameter counts for CNN vs MLP architectures
"""
import torch
from core.networks import PolicyNet, ValueNet, FlexibleNet


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    input_dim = 76
    action_dim = 4

    print("=" * 60)
    print("Parameter Count Verification")
    print("=" * 60)
    print()

    # CNN Architecture
    print("CNN Architecture (CardAwareNet):")
    print("-" * 60)

    cnn_policy = PolicyNet(
        input_dim=input_dim,
        output_dim=action_dim,
        hidden_layers=[256, 256, 128],
        use_layer_norm=True,
        card_encoder_channels=[16, 32],
        card_embedding_dim=128,
    )

    cnn_value = ValueNet(
        input_dim=input_dim,
        hidden_layers=[256, 256, 128],
        use_layer_norm=True,
        card_encoder_channels=[16, 32],
        card_embedding_dim=128,
    )

    cnn_policy_params = count_params(cnn_policy)
    cnn_value_params = count_params(cnn_value)

    print(f"  PolicyNet: {cnn_policy_params:,} parameters")
    print(f"  ValueNet:  {cnn_value_params:,} parameters")
    print(f"  Total:     {cnn_policy_params + cnn_value_params:,} parameters")
    print()

    # MLP Architecture
    print("MLP Architecture (FlexibleNet):")
    print("-" * 60)

    mlp_policy = FlexibleNet(
        input_dim=input_dim,
        output_dim=action_dim,
        hidden_layers=[512, 400, 256],
        use_layer_norm=True,
    )

    mlp_value = FlexibleNet(
        input_dim=input_dim,
        output_dim=1,
        hidden_layers=[512, 400, 256],
        use_layer_norm=True,
    )

    mlp_policy_params = count_params(mlp_policy)
    mlp_value_params = count_params(mlp_value)

    print(f"  PolicyNet: {mlp_policy_params:,} parameters")
    print(f"  ValueNet:  {mlp_value_params:,} parameters")
    print(f"  Total:     {mlp_policy_params + mlp_value_params:,} parameters")
    print()

    # Comparison
    print("Comparison:")
    print("-" * 60)
    cnn_total = cnn_policy_params + cnn_value_params
    mlp_total = mlp_policy_params + mlp_value_params
    diff = abs(cnn_total - mlp_total)
    diff_pct = (diff / cnn_total) * 100

    print(f"  CNN Total: {cnn_total:,} parameters")
    print(f"  MLP Total: {mlp_total:,} parameters")
    print(f"  Difference: {diff:,} parameters ({diff_pct:.1f}%)")
    print()

    if diff_pct < 5:
        print("✓ Parameter counts are comparable (< 5% difference)")
    else:
        print("⚠ Parameter counts differ by more than 5%")

    print("=" * 60)


if __name__ == "__main__":
    main()
