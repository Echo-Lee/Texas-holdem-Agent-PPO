---
title: Texas Hold'em PPO Agent
emoji: 🎰
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# 🎰 Texas Hold'em PPO Agent

Play heads-up Texas Hold'em poker against AI agents trained with Proximal Policy Optimization (PPO)!

## 🎮 Features

- **Multiple AI Opponents**: Choose from CNN or MLP-based agents at different training stages
- **Real-time Game Info**: Track pot size and bet amounts
- **Professional UI**: Casino-themed dark interface
- **Fair Play**: Watch agent actions and see cards in real poker order

## 🤖 Available Models

- **Stage 1**: Early training agents (vs rule-based opponents)
- **Stage 2**: Improved agents (fine-tuned vs frozen agents)
- **Stage 3**: Final agents (advanced fine-tuning)

Each stage available in both:
- **CNN**: Card-aware convolutional network (~357k params)
- **MLP**: Pure multi-layer perceptron (~120k params)

## 🎯 How to Play

1. Select an opponent from the dropdown
2. Set your starting chips
3. Click "🎲 New Match" to begin
4. Use action buttons: Call, Raise, Fold, or Check
5. Try different opponents to test various strategies!

## 🧠 Architecture

Trained using PPO with a three-stage curriculum:
- Stage 1: Multiple agents vs rule-based opponents
- Stage 2: Fine-tuning with replay buffer
- Stage 3: Advanced refinement

See the [GitHub repository](https://github.com/your-username/your-repo) for training details and source code.

## 📊 Model Comparison

Compare CNN vs MLP architectures to see which performs better:
- CNN uses card-structure inductive bias
- MLP processes features directly
- Both trained with identical hyperparameters

---

Built with [PettingZoo](https://pettingzoo.farama.org/), [PyTorch](https://pytorch.org/), and [Gradio](https://gradio.app/)
