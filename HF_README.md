---
title: Texas Hold'em PPO Agent
emoji: 🎰
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🎰 Texas Hold'em PPO Agent

Play heads-up Texas Hold'em against the **final tuned stage3 PPO agent** from this repository.

## Features

- **Stage3 tuned checkpoint** loaded by default in the Hugging Face UI
- **Dark casino table UI** built with Gradio
- **Persistent match tracking** across hands
- **Full card display** for hero, board, and revealed opponent cards at showdown

## About the AI

This Space uses the repo's strongest saved checkpoint:
- **Model**: `models/stage3_final_policy.pt`
- **Environment**: PettingZoo `texas_holdem_v4`
- **Algorithm**: PPO with GAE
- **Network**: card-aware policy network with a CNN card encoder

The stage3 tuned run reached:
- **Final average reward**: `5.380`
- **Best average reward**: `6.840`
- **Final win rate**: `0.721`

## How to Play

1. Click **New Match** to start a match with the selected chip stack.
2. Click **Deal Hand** to begin the next hand.
3. Use the action buttons: **Call**, **Raise**, **Fold**, or **Check**.
4. The action buttons enable only when that move is legal.
5. Review the hand log and chip display as the match progresses.

## Notes

- The UI now loads the stage3 tuned policy by default.
- If you ever want to override the model on Hugging Face Spaces, set the `MODEL_PATH` environment variable.

## Links

- [Repository README](./README.md)
- [Training entry point](./core/main.py)
- [Network definition](./core/networks/policy_value_network.py)
