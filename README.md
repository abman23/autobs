# AutoBS: Autonomous Base Station Deployment Framework with Reinforcement Learning and Digital Twin Network
[![arXiv](https://img.shields.io/badge/arXiv-2502.19647-b31b1b.svg)](https://arxiv.org/abs/2502.19647)

This repository contains the official implementation of **“AutoBS: Autonomous Base Station Deployment Framework with Reinforcement Learning and Digital Twin Network”** presented at **ICML'25 (ML4Wireless Workshop)** (arXiv:2502.19647, 2025).

## Demo

<div>
  <img src="figures/animation_autobs.gif" alt="AutoBS deployment animation" width="500" />
</div>

## Highlights

- **DRL-based deployment** – novel reinforcement‑learning framework for single and multi‑base‑station (BS) deployment that incorporates [PMNet](https://arxiv.org/abs/2312.03950) for real‑time, site‑specific channel prediction:contentReference[oaicite:1]{index=1}.
- **Milliseconds inference** – reduces deployment inference time from hours to milliseconds compared with exhaustive methods, enabling practical large‑scale optimisation:contentReference[oaicite:2]{index=2}.
- **Pre‑trained models included** – checkpoints for the single‑BS agent, multi‑BS agent and PMNet are provided for easy reproduction:contentReference[oaicite:3]{index=3}.
- **Visualisation** – uses SionnaRT to visualise deployment outcomes:contentReference[oaicite:4]{index=4}.

## Overview

AutoBS integrates a digital twin network with deep reinforcement learning to autonomously determine optimal base station locations. By combining PMNet’s channel prediction with a digital twin model of the radio environment, the framework learns to balance coverage and capacity objectives and adapts to both single‑ and multi‑BS deployment scenarios. This design enables real‑time optimisation of dense networks on commodity hardware.

## Quick Start

### 1. Clone
```sh
git clone https://github.com/abman23/autobs.git
cd autobs
````

### 2. Environment

Create and activate a Python environment (tested with Python 3.10):

```sh
conda create -n autobs_env python=3.10
conda activate autobs_env
pip install -r requirements.txt
```

### 3. Available checkpoints

Pre‑trained checkpoints are provided for convenience:

| Model           | Download Link |
| --------------- | ------------- |
| Single‑BS Agent | [Download]()  |
| Multi‑BS Agent  | [Download]()  |
| PMNet           | [Download]()  |

### 4. Inference

To evaluate AutoBS on a test map:

```sh
python inference.py \
    --version [single/multi] \
    --crop_id [0-15] \
    --reward_type [coverage/capacity]
```

After running, the output coverage map is saved under `visualize/sionna_output/`.

## Citation

```
@article{lee2025autobs,
    title   = {AutoBS: Autonomous Base Station Deployment Framework with Reinforcement Learning and Digital Twin Network},
    author  = {Ju-Hyung Lee and Andreas F. Molisch},
    year    = {2025},
    journal = {arXiv preprint arXiv:2502.19647},
}
```

## Contributors

We acknowledge the contributions of Arjun Balamwar and Yanqing Lu to the framework design and simulations.
