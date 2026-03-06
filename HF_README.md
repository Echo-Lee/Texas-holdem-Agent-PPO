---
title: Texas Hold'em PPO Agent
emoji: 🎰
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: mit
---

# 🎰 Texas Hold'em PPO Agent

Play heads-up Texas Hold'em poker against an AI agent trained with **Proximal Policy Optimization (PPO)** reinforcement learning!

## 🎮 Features

- **Dark Casino Theme**: Beautiful poker table interface with professional styling
- **Real-time Gameplay**: Instant AI responses powered by PPO neural networks
- **Opponent Modeling**: The AI predicts your strategy to make better decisions
- **Self-Play Trained**: Agent improved by playing millions of hands against itself
- **Match Tracking**: Full game history and chip stack visualization

## 🤖 About the AI

This agent was trained using:
- **Algorithm**: Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)
- **Opponent Modeling**: Supervised learning to predict opponent actions
- **Self-Play**: Trained against evolving versions of itself
- **Architecture**: Deep neural networks with LayerNorm for stable training

The agent has learned advanced poker strategies through reinforcement learning, including:
- Bluffing and semi-bluffing
- Pot odds calculation
- Position awareness
- Bet sizing for value and protection

## 🎯 How to Play

1. Click **"Start New Match"** to begin with your starting chip stack
2. Click **"Deal New Hand"** to start each new hand
3. Choose your action: **Call**, **Raise**, **Fold**, or **Check**
4. Try to outplay the AI and build your chip stack!

## 📊 Technical Details

- **Environment**: PettingZoo texas_holdem_v4
- **Training Framework**: PyTorch
- **UI Framework**: Gradio with custom casino theme
- **Metrics Tracked**: Win rate, average reward, opponent model accuracy

## 🔗 Links

- [GitHub Repository](https://github.com/Echo-Lee/Leduc-holdem-Agent-PPO)
- [Training Code](https://github.com/Echo-Lee/Leduc-holdem-Agent-PPO/blob/main/core/main.py)
- [Model Architecture](https://github.com/Echo-Lee/Leduc-holdem-Agent-PPO/blob/main/core/networks/policy_value_network.py)

## 🙏 Acknowledgments

Built with [PettingZoo](https://pettingzoo.farama.org/), [Gradio](https://gradio.app/), and [PyTorch](https://pytorch.org/).

Inspired by DeepMind's AlphaGo and OpenAI's poker research.

---

**Good luck at the tables!** 🍀
