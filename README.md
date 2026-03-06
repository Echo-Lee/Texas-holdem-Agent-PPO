# Texas Hold'em PPO Agent

Reinforcement Learning project implementing **PPO (Proximal Policy Optimization)** for heads-up Texas Hold'em poker using PettingZoo. Features opponent modeling, self-play training, and a Gradio UI for playing against the trained agent.

Play the cards on this URL: https://huggingface.co/spaces/ChenyuEcho/texas-holdem-ppo

## 🎯 Features

### Training & AI
- **PPO Algorithm** with GAE (Generalized Advantage Estimation)
- **Opponent Modeling** - Predicts opponent actions to improve decision-making
- **Self-Play Training** - Agent improves by playing against past versions of itself
- **Rule-Based Opponents** - Train against aggressive, conservative, or random strategies
- **W&B Integration** - Track training metrics with Weights & Biases

### UI & Deployment
- **🎰 Dark Casino Theme** - Professional poker table interface with green felt and gold accents
- **Glassmorphism Effects** - Modern translucent panels and smooth animations
- **Responsive Design** - Beautiful card displays with shadows and hover effects
- **Real-time Gameplay** - Instant AI responses with full action history
- **One-Click Deployment** - Ready for Hugging Face Spaces

## 🐛 Recent Bug Fixes (2026-03-06)

Fixed dimension mismatch comments and incorrect game naming:
- **runner.py** - Updated stale comments from Leduc Hold'em dimensions (40/36) to Texas Hold'em (76/72)
- **main.py** - Changed wandb tag from "Leduc" to "Texas-Holdem"
- **rule_based_agent.py** - Fixed comment referencing wrong game

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Leduc-holdem-Agent-PPO.git
cd Leduc-holdem-Agent-PPO

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# or: .venv\Scripts\activate.bat  # Windows cmd
# or: .venv/Scripts/Activate.ps1  # PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with current configuration
python core/main.py
```

Configuration is managed through `running_config.yaml`:
- `system.use_opponent_model` - Enable/disable opponent prediction network
- `selfplay.enabled` - Toggle self-play vs. static opponent
- `opponent.style` - "random", "aggressive", or "conservative"
- `train.*` - PPO hyperparameters (learning rate, batch size, etc.)
- `model.*` - Network architecture settings

### Playing Against Your Agent

```bash
# Local play with Gradio UI
python core/play_game.py

# Or use the deployment-ready version
python app.py
```

**Note**: Update `MODEL_PATH` in the script to point to your trained model.

## 📊 Architecture

### Core Components

**Agent** (`core/agents/`)
- `BaseRLAgent` - Abstract base class with action selection and update interface
- `PPOAgent` - PPO implementation with clipped surrogate loss
- `RuleBasedAgent` - Deterministic strategies for baseline comparison

**Networks** (`core/networks/`)
- `PolicyNet` - Outputs action probability distribution
- `ValueNet` - State value estimation
- `OpponentPredictor` - Supervised learning to predict opponent actions

**Environment** (`core/environments/`)
- `Runner` - Manages PettingZoo texas_holdem_v4 environment
- Collects trajectories for both player and opponent
- Trains opponent model from observed actions

### Observation Processing

Agent observation (76-dim):
- **72-dim**: Raw game state from texas_holdem_v4 (card encoding + betting info)
- **4-dim**: Opponent action probabilities (from opponent predictor, or zeros if disabled)

### Training Modes

**Self-Play** (`selfplay.enabled=True`):
- Opponent mirrors the learning agent's networks
- Periodically updated every N iterations
- Opponent receives "blind" observations (no opponent predictions)

**Static Opponent** (`selfplay.enabled=False`):
- Rule-based opponent with fixed strategy
- Used for baseline evaluation and ablation studies

## 🌐 Deployment

### Hugging Face Spaces (Recommended)

Perfect for ML apps with large dependencies. **No size limits, free hosting!**

**Quick Deploy:**
```bash
# Use automated script
chmod +x deploy_to_hf.sh
./deploy_to_hf.sh YOUR_USERNAME texas-holdem-ppo
```

**Manual Deploy:**
1. Create a Space at https://huggingface.co/new-space (select Gradio SDK)
2. Clone your Space and copy files
3. Push to deploy

Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE`

**See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed step-by-step instructions.**

## 📁 Project Structure

```
d:\RL Project/
├── core/
│   ├── agents/              # RL agents and rule-based opponents
│   ├── environments/        # Environment runner
│   ├── networks/            # Neural network architectures
│   └── utils/               # Utility functions
├── models/                  # Saved model checkpoints
├── poker_cards/             # Card images for UI
├── results/                 # Training plots and metrics
├── wandb/                   # W&B experiment tracking
├── core/main.py             # Training script
├── core/play_game.py        # Interactive gameplay UI
├── app.py                   # Dark casino UI (deployment ready)
├── running_config.yaml      # Training configuration
├── requirements.txt         # Python dependencies
├── deploy_to_hf.sh          # Automated HF Spaces deployment
├── HF_README.md             # Hugging Face Space description
└── DEPLOYMENT.md            # Deployment guide
```

## 🎮 Action Space

Texas Hold'em actions:
- `0`: Call
- `1`: Raise
- `2`: Fold
- `3`: Check

## 📈 Training Details

- **Algorithm**: PPO with clipped surrogate loss
- **Value Training**: Separate value network with MSE loss
- **Advantages**: GAE (λ=0.95) recomputed after value updates
- **Gradient Clipping**: max_norm=0.5 for both policy and value
- **Entropy Regularization**: Encourages exploration
- **Action Masking**: Invalid actions masked with -1e9 logits

## 🔧 Configuration Parameters

Key settings in `running_config.yaml`:

```yaml
system:
  device: "auto"              # "cuda", "cpu", or "auto"
  use_opponent_model: True

train:
  episodes_per_batch: 256
  num_iterations: 100
  mini_batch_size: 64
  lr: 0.0003
  gamma: 0.99

ppo:
  clip_param: 0.2
  policy_epochs: 5
  value_epochs: 5
  gae_lambda: 0.95
  entropy_coef: 0.01

selfplay:
  enabled: True
  update_every: 20            # Update opponent every N iterations
```

## 📊 Monitoring

Training metrics logged to W&B:
- Average reward per episode
- Win rate against opponent
- Opponent model loss and accuracy
- Policy loss, value loss, entropy (optional)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Built with [PettingZoo](https://pettingzoo.farama.org/) for multi-agent environments
- Uses [Gradio](https://gradio.app/) for the web interface
- Inspired by DeepMind's AlphaGo and OpenAI's poker research
