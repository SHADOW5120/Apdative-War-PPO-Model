# Apdative-War-PPO-Model
NPC Tactical AI — PPO Training

PPO-based Reinforcement Learning training script for a 2D Isometric combat NPC. Exports an `.onnx` model for **Unity Sentis** inference.

---

## Requirements

```bash
pip install torch numpy onnx matplotlib
```

## Quick Start

```bash
python train.py
```

**Output files:**

| File | Description |
|---|---|
| `npc_tactical_iso2d_best.onnx` | Best model → import into Unity |
| `npc_tactical_iso2d_best.pt` | PyTorch checkpoint |
| `action_mapping.json` | Action ID mapping for C# |
| `training_log.csv` | Full training log |
| `training_curves.png` | Training charts |

---

## Architecture

**Observation** (6 dims): `[agent_hp, dx, dy, dist, enemy_hp, in_range]`

**Actions** (10): `Idle` · `Move N/NE/E/SE/S/SW/W/NW` · `Attack`

**Network**: Actor-Critic — 2× hidden layers (64 units, Tanh)

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Total steps | 1,000,000 |
| Learning rate | 3e-4 (linear anneal) |
| Gamma / GAE λ | 0.99 / 0.95 |
| Clip coefficient | 0.2 |
| Early stop KL | 0.02 |
| Rollout length | 2048 |
| PPO epochs | 4 |
| Mini-batch size | 128 |

---

## Environment

- 2D map `[-8, 8] × [-8, 8]`, step size `0.5` units, episode limit `100` steps
- Reward: `+2` hit · `+20` kill · `-5` death · potential-based shaping toward enemy
- Episode ends on: kill · death · timeout

---

## Unity Integration

1. Copy `npc_tactical_iso2d_best.onnx` → `Assets/Resources/`
2. Match `action_mapping.json` with your C# action enum
3. Run inference with **Unity Sentis**
