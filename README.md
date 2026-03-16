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

## Recent Updates (2026-03-15)

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

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Training

Run the default three-stage pipeline:

```bash
python core/main.py --config running_config.yaml
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

Local Gradio UI:

```bash
python core/play_game.py
```

Deployment-ready app:

```bash
python app.py
```

By default:
- `app.py` loads `models/stage3_final_policy.pt`
- `core/play_game.py` still points to `models/stage2_final_policy.pt`
- the strongest saved checkpoint in this repo is currently `models/stage3_final_policy.pt`

To launch the app with the stage3 model in PowerShell:

```powershell
$env:MODEL_PATH="models/stage3_final_policy.pt"
python app.py
```

For `core/play_game.py`, update the `MODEL_PATH` constant if you want the local UI to use stage3 as well.

## Config Files

`running_config.yaml` is the **source-of-truth config** for the current codebase.

That file is used by default by:
- `core/main.py`
- `random_search.py`
- `app.py`
- `core/play_game.py`

The `run_config.yaml` files under `results/` are **saved run snapshots**, not the active config:
- `results/run_config.yaml` matches the older baseline run stored in `results/`
- `results/random_search/final/run_config.yaml` matches the tuned final run stored in `results/random_search/final/`

Those snapshot files are still useful because they preserve the exact settings used for each result folder, but they are only used if you explicitly point a script to them.

If you want the config to edit for future runs, edit:

```text
running_config.yaml
```

If you want the exact config that produced a saved result, open the `run_config.yaml` inside that result directory.

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
- `PolicyNet` - card-aware policy network
- `ValueNet` - card-aware value network
- `OpponentPredictor` - optional supervised opponent action model

**Environment** (`core/environments/`)
- `Runner` - batch collection for PettingZoo `texas_holdem_v4`
- handles action masking, opponent interaction, reward tracking, and opponent-model training

**Replay utilities** (`core/utils/`)
- `ReplayBatchBuffer` - stores and reuses prior training batches during stage2/stage3
- plotting and summary helpers for CSV/JSON output

### Observation Processing

Agent observation dimension: **76**
- **72 dims**: raw `texas_holdem_v4` observation
- **4 dims**: opponent-action prediction features

When `system.use_opponent_model: false` (the current default), those final 4 features are zero-padded.

### CNN Card Encoder

The policy and value networks in `core/networks/policy_value_network.py` use a dedicated **CNN card encoder** before the main MLP.

Card-processing flow:
- the first **52** observation entries are treated as card-presence bits
- those bits are reshaped into a **`13 x 4` rank-by-suit matrix**
- the matrix is passed through stacked **`3x3` Conv2d + ReLU** layers
- the convolution output is flattened and projected into a dense card embedding
- that embedding is concatenated with the remaining non-card features before the final MLP backbone

Current default encoder settings:

```yaml
model:
  agent:
    card_encoder_channels: [16, 32]
    card_embedding_dim: 128
```

Why this helps:
- the reshaped matrix gives the network a structured view of cards instead of treating all 52 bits as unrelated inputs
- shared convolution filters can detect local rank-suit patterns such as same-suit groupings and nearby-rank structures
- the learned card embedding compresses sparse card indicators into a denser representation before policy/value prediction

Conceptually, the encoder turns:

```text
52 card bits -> 13 x 4 matrix -> conv layers -> dense card embedding -> MLP -> policy/value head
```

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

Main settings live in `running_config.yaml`.

Current defaults include:

```yaml
system:
  device: "auto"
  use_opponent_model: false

model:
  agent:
    hidden_layers: [256, 256, 128]
    use_layer_norm: true
    card_encoder_channels: [16, 32]
    card_embedding_dim: 128

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

random_search:
  trials_per_stage: 10
```

Useful knobs:
- `system.use_opponent_model` - enable or disable opponent prediction features
- `model.agent.*` - policy/value architecture
- `stages.stage1.*` - seed-agent training setup
- `stages.stage2.replay.*` - replay mix for stage2
- `stages.stage3.replay.*` - replay mix for stage3
- `save.*` - exported checkpoint names in `models/`

## Deployment

### Hugging Face Spaces

Quick deploy:

```bash
chmod +x deploy_to_hf.sh
./deploy_to_hf.sh YOUR_USERNAME texas-holdem-ppo
```

Manual deploy:
1. Create a new Gradio Space on Hugging Face
2. Copy the repository files into the Space
3. Push the Space repo

See `DEPLOYMENT.md` for the deployment checklist.

## Project Structure

```text
Texas-holdem-Agent-CNNPPO/
|-- core/
|   |-- agents/
|   |-- environments/
|   |-- networks/
|   `-- utils/
|-- models/
|   |-- stage2_final_policy.pt
|   `-- stage3_final_policy.pt
|-- poker_cards/
|-- results/
|   |-- random_search/
|   `-- ...
|-- app.py
|-- core/main.py
|-- core/play_game.py
|-- random_search.py
|-- running_config.yaml
|-- requirements.txt
|-- HF_README.md
`-- DEPLOYMENT.md
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
