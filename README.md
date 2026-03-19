# Texas Hold'em PPO Agent

Reinforcement learning project implementing **PPO (Proximal Policy Optimization)** for heads-up Texas Hold'em poker using PettingZoo. The current codebase trains agents through a **three-stage curriculum**, supports **stage-wise random search**, and includes **Gradio UIs** for local play and Hugging Face Spaces deployment.

Play the cards on this URL: https://holdemagent.lovable.app/

## Features

### Training & AI
- **Three-stage PPO pipeline** for progressively stronger agents
- **Card-aware policy/value networks** with a CNN-style card encoder
- **Replay-assisted fine-tuning** in later stages
- **Sequential random search** for stage-specific hyperparameters
- **Optional opponent modeling** with a separate predictor network
- **Rule-based opponents** for baseline stage training
- **W&B integration** for experiment logging when enabled

### UI & Deployment
- **Dark casino Gradio app** in `app.py`
- **Local match UI** in `core/play_game.py`
- **Card image assets** for full table rendering
- **Hugging Face Spaces ready** deployment flow

## Recent Updates (2026-03-19)

- Added **MLP baseline comparison** with configurable pure MLP architecture (no CNN encoder)
- Implemented **agent battle system** (`scripts/agent_battle.py`) for comparing different model architectures
- Separate training configs: `running_config.yaml` (CNN) and `running_config_mlp.yaml` (MLP)
- Organized utility scripts into `scripts/` directory for cleaner project structure

## Previous Updates (2026-03-15)

- Replaced the older single-stage/self-play README flow with the current **stage1 -> stage2 -> stage3** training pipeline implemented in `core/main.py`
- Added **`random_search.py`** for sequential stage-wise hyperparameter tuning
- Added tuned outputs under **`results/random_search/`** and a final stage3 checkpoint at **`models/stage3_final_policy.pt`**
- Switched the default model architecture to a **card-aware network** with configurable convolution channels and embedding size
- Added **replay buffer support** for fine-tuning stages
- Updated the deployment app to load model/config from **`MODEL_PATH`** and **`CONFIG_PATH`** environment variables

## Quick Start

### Installation

```bash
git clone <your-repo-url>
cd Texas-holdem-Agent-CNNPPO

# Create virtual environment
py -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Training

Run the default three-stage pipeline (CNN):

```bash
python -m core.main --config running_config.yaml
```

Run with MLP architecture:

```bash
python -m core.main --config running_config_mlp.yaml
```

Run sequential random search:

```bash
python random_search.py --config running_config.yaml
```

Tune only up to a specific stage:

```bash
python random_search.py --config running_config.yaml --stage stage2
```

### Playing Against Your Agent

Launch the Gradio web interface:

```bash
python app.py
```

**Features:**
- Select any model from `models/` directory via dropdown
- Real-time Pot and To Call display
- Automatic CNN/MLP architecture detection
- Track raise amounts in action log
- Casino-themed dark UI

The app automatically detects the model architecture (CNN or MLP) from the filename and loads the appropriate configuration.

### Agent vs Agent Battle

Compare different architectures or training stages:

```bash
python scripts/agent_battle.py --agent1 models/stage3_final_policy_cnn.pt --agent2 models/stage3_final_policy_mlp.pt --games 1000
```

This will run 1000 games and report win rates and average rewards for both agents.

## Config Files

Two main configuration files:

**`running_config.yaml`** - CNN architecture (card-aware network)
- Model type: `cnn`
- Uses `CardMatrixEncoder` for card features
- Results saved to `results/`
- Models saved as `*_cnn.pt`

**`running_config_mlp.yaml`** - Pure MLP architecture
- Model type: `mlp`
- Direct MLP processing of all features
- Results saved to `results_mlp/`
- Models saved as `*_mlp.pt`

Both configs are used by:
- `core/main.py` (specify with `--config`)
- `random_search.py` (specify with `--config`)
- `app.py` (auto-detects from model filename)
- `scripts/agent_battle.py` (auto-detects from model filename)

The `run_config.yaml` files under `results/` are **saved run snapshots**, not the active config:
- `results/run_config.yaml` matches the older baseline run stored in `results/`
- `results/random_search/final/run_config.yaml` matches the tuned final run stored in `results/random_search/final/`

Those snapshot files are still useful because they preserve the exact settings used for each result folder, but they are only used if you explicitly point a script to them.

## Architecture

### Core Components

**Training entry points**
- `core/main.py` - full multi-stage training pipeline
- `random_search.py` - stage-wise hyperparameter search and final tuned run

**Agents** (`core/agents/`)
- `BaseRLAgent` - shared agent interface
- `PPOAgent` - PPO learner with clipped objective and GAE
- `RuleBasedAgent` - aggressive, conservative, or random baseline policies

**Networks** (`core/networks/`)
- `PolicyNet` - card-aware policy network (CNN) or pure MLP
- `ValueNet` - card-aware value network (CNN) or pure MLP
- `FlexibleNet` - configurable MLP for baseline comparison
- `OpponentPredictor` - optional supervised opponent action model

**Environment** (`core/environments/`)
- `Runner` - batch collection for PettingZoo `texas_holdem_v4`
- handles action masking, opponent interaction, reward tracking, and opponent-model training

**Replay utilities** (`core/utils/`)
- `ReplayBatchBuffer` - stores and reuses prior training batches during stage2/stage3
- plotting and summary helpers for CSV/JSON output

**Utility scripts** (`scripts/`)
- `agent_battle.py` - compare two trained agents head-to-head
- `test_implementation.py` - verify CNN/MLP implementations
- `verify_params.py` - check parameter counts for architectures

### Observation Processing

Agent observation dimension: **76**
- **72 dims**: raw `texas_holdem_v4` observation
- **4 dims**: opponent-action prediction features

When `system.use_opponent_model: false` (the current default), those final 4 features are zero-padded.

### Network Architectures

The project supports two network architectures for comparison:

#### CNN Architecture (Card-Aware)

The policy and value networks in `core/networks/policy_value_network.py` use a dedicated **CNN card encoder** before the main MLP.

Card-processing flow:
- the first **52** observation entries are treated as card-presence bits
- those bits are reshaped into a **`13 x 4` rank-by-suit matrix**
- the matrix is passed through stacked **`3x3` Conv2d + ReLU** layers
- the convolution output is flattened and projected into a dense card embedding
- that embedding is concatenated with the remaining non-card features before the final MLP backbone

Current default encoder settings (in `running_config.yaml`):

```yaml
system:
  model_type: "cnn"

model:
  agent:
    card_encoder_channels: [16, 32]
    card_embedding_dim: 128
    hidden_layers: [256, 256, 128]
```

**Parameter count**: ~357k parameters
- CardMatrixEncoder: 218k (Conv + projection)
- MLP backbone: 139k

Why this helps:
- the reshaped matrix gives the network a structured view of cards instead of treating all 52 bits as unrelated inputs
- shared convolution filters can detect local rank-suit patterns such as same-suit groupings and nearby-rank structures
- the learned card embedding compresses sparse card indicators into a denser representation before policy/value prediction

Conceptually, the encoder turns:

```text
52 card bits -> 13 x 4 matrix -> conv layers -> dense card embedding -> MLP -> policy/value head
```

#### MLP Architecture (Pure MLP Baseline)

A simpler baseline using `FlexibleNet` that directly processes all 76 observation dimensions:

Current default settings (in `running_config_mlp.yaml`):

```yaml
system:
  model_type: "mlp"

model:
  agent:
    mlp_hidden_layers: [256, 256, 128]
    use_layer_norm: true
```

**Parameter count**: ~120k parameters
- All layers process features directly
- No structural bias on card relationships

```text
76 dims -> MLP layers -> policy/value head
```

This architecture serves as a baseline to evaluate whether the CNN's card-structure inductive bias actually helps for poker.

### Training Stages

**Stage 1**
- trains `num_agents` independent PPO agents against a rule-based opponent
- current default: `2` agents, `20` iterations, `64` episodes per batch

**Stage 2**
- fine-tunes one stage1 agent against a frozen stage1 opponent
- mixes current batches with replayed prior batches
- current default: `80` iterations, `256` episodes per batch

**Stage 3**
- fine-tunes the stage2 agent again against a frozen earlier-stage opponent
- keeps replay enabled for additional stabilization
- current default: `60` iterations, `256` episodes per batch

## Current Results

### Baseline pipeline (`results/`)

This is an older baseline run snapshot.

- `stage1_agent_1`: final avg reward `2.785`, final win rate `0.945`
- `stage1_agent_2`: final avg reward `2.426`, final win rate `0.938`
- `stage2`: final avg reward `2.717`, final win rate `0.588`

### Tuned random-search pipeline (`results/random_search/final/`)

This is the most complete result set for the current code structure, including stage3.

- `stage1_agent_1`: final avg reward `3.323`, final win rate `0.969`
- `stage1_agent_2`: final avg reward `2.584`, final win rate `0.906`
- `stage2`: final avg reward `3.992`, final win rate `0.686`
- `stage3`: final avg reward `5.380`, best avg reward `6.840`, final win rate `0.721`

Best tuned parameters currently saved in `results/random_search/best_params.json`:

```json
{
  "stage1": {
    "lr": 0.00022605,
    "mini_batch_size": 32,
    "gamma": 0.99071,
    "entropy_coef": 0.01208
  },
  "stage2": {
    "lr": 1.904e-05,
    "mini_batch_size": 128,
    "replay_every": 6,
    "replay_start_iteration": 8,
    "max_prev_stage_batches": 8,
    "max_current_stage_batches": 4
  },
  "stage3": {
    "lr": 1.407e-05,
    "mini_batch_size": 128,
    "replay_every": 5,
    "replay_start_iteration": 4,
    "max_prev_stage_batches": 4,
    "max_current_stage_batches": 6
  }
}
```

## Action Space

Texas Hold'em actions:
- `0`: Call
- `1`: Raise
- `2`: Fold
- `3`: Check

## Training Details

- **Algorithm**: PPO with clipped surrogate loss
- **Advantages**: GAE with `gae_lambda=0.95`
- **Action masking**: invalid actions are masked before sampling/updating
- **Gradient clipping**: `max_norm=0.5` for policy and value networks
- **Entropy regularization**: controlled by `ppo.entropy_coef`
- **Card encoder**: first 52 observation bits are reshaped into a `13 x 4` rank-by-suit matrix and encoded with stacked `3x3` convolutions before the MLP backbone

## Configuration

### CNN Configuration (`running_config.yaml`)

```yaml
system:
  device: "auto"
  model_type: "cnn"
  use_opponent_model: false

model:
  agent:
    hidden_layers: [256, 256, 128]
    use_layer_norm: true
    card_encoder_channels: [16, 32]
    card_embedding_dim: 128

results:
  dir: "results"

save:
  stage1_policy: "models/stage1_agent_{agent_id}_cnn_policy.pt"
  stage2_policy: "models/stage2_final_cnn.pt"
  stage3_policy: "models/stage3_final_cnn.pt"

stages:
  stage1:
    num_agents: 2
    iterations: 20
    episodes_per_batch: 64
    opponent_style: "random"

  stage2:
    iterations: 80
    episodes_per_batch: 256
    mini_batch_size: 128

  stage3:
    iterations: 60
    episodes_per_batch: 256
    mini_batch_size: 128
```

### MLP Configuration (`running_config_mlp.yaml`)

```yaml
system:
  device: "auto"
  model_type: "mlp"
  use_opponent_model: false

model:
  agent:
    mlp_hidden_layers: [256, 256, 128]
    use_layer_norm: true

results:
  dir: "results_mlp"

save:
  stage1_policy: "models/stage1_agent_{agent_id}_mlp_policy.pt"
  stage2_policy: "models/stage2_final_mlp.pt"
  stage3_policy: "models/stage3_final_mlp.pt"

# stages configuration same as CNN
```

### Useful Configuration Knobs

- `system.model_type` - "cnn" or "mlp" architecture
- `system.use_opponent_model` - enable or disable opponent prediction features
- `model.agent.*` - policy/value architecture
- `results.dir` - where to save training results
- `stages.stage1.*` - seed-agent training setup
- `stages.stage2.replay.*` - replay mix for stage2
- `stages.stage3.replay.*` - replay mix for stage3
- `save.*` - exported checkpoint names in `models/`

## Deployment

### Hugging Face Spaces

Manual deploy:
1. Create a new Gradio Space on Hugging Face
2. Copy `app.py`, `requirements.txt`, `poker_cards/`, `models/`, and `core/` to the Space
3. Set the model path in the Space's environment variables if needed
4. Push the Space repo

The app will automatically detect the model architecture (CNN or MLP) from the checkpoint filename.

## Project Structure

```text
Texas-holdem-Agent-CNNPPO/
|-- core/
|   |-- agents/          # PPO, rule-based agents
|   |-- environments/    # PettingZoo runner
|   |-- networks/        # Policy/Value networks (CNN & MLP)
|   |-- utils/           # Replay buffer, plotting
|   `-- main.py          # Main training pipeline
|
|-- scripts/
|   |-- agent_battle.py        # Agent comparison tool
|   |-- test_implementation.py # Architecture tests
|   `-- verify_params.py       # Parameter counting
|
|-- models/
|   |-- stage2_final_policy_cnn.pt
|   |-- stage3_final_policy_cnn.pt
|   |-- stage2_final_policy_mlp.pt
|   `-- stage3_final_policy_mlp.pt
|
|-- poker_cards/         # Card image assets
|
|-- results/             # CNN training results
|   |-- stage1_agent_1/
|   |-- stage2/
|   `-- stage3/
|
|-- results_mlp/         # MLP training results
|   |-- stage1_agent_1/
|   |-- stage2/
|   `-- stage3/
|
|-- app.py               # Deployment Gradio app
|-- random_search.py     # Hyperparameter tuning
|-- running_config.yaml      # CNN config
|-- running_config_mlp.yaml  # MLP config
|-- requirements.txt
`-- README.md
```

## Monitoring

Training outputs are written as:
- `metrics.csv` per stage
- `summary.json` per stage
- `run_summary.json` per training run
- reward/win-rate plots as `.png`

W&B logging is supported, but the current config keeps it disabled by default:

```yaml
wandb:
  enabled: false
  mode: "offline"
```

## Contributing

Contributions are welcome. Please open an issue or pull request if you want to improve the training pipeline, evaluation, or UI.

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Built with [PettingZoo](https://pettingzoo.farama.org/) for the poker environment
- Uses [PyTorch](https://pytorch.org/) for training
- Uses [Gradio](https://gradio.app/) for the web interface
