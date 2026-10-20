# World Model — CarRacing (partial reimplementation)

A small, from-scratch PyTorch reimplementation of the **World Models** architecture
from Ha & Schmidhuber, 2018 — [arXiv:1803.10122](https://arxiv.org/pdf/1803.10122) —
trained on OpenAI Gym's `CarRacing-v3` environment.

The paper factorizes an agent into three parts: a **V**ision model (VAE), a
**M**emory/predictive model (MDN-RNN), and a **C**ontroller (a small linear policy
optimized with CMA-ES inside the learned "dream"). This repo currently implements
and trains **V** and **M**, plus the data-collection pipeline that feeds them. The
controller is not implemented yet.

## Relation to the 2018 paper

| Paper component | Repo status | Where |
| --- | --- | --- |
| Data collection from a random policy | Done | `envs/collect_rollouts.py` |
| **V** — convolutional VAE, 64×64 RGB → 32-d latent `z` | Done | `vmodel/` |
| **M** — MDN-RNN predicting `p(z_{t+1} | z_t, a_t)` with K mixture components | Model + loss done, training loop is a smoke test | `mmodel/` |
| **C** — linear controller trained with CMA-ES in the dream | Not implemented | — |

Design choices follow the paper:

- The VAE is trained unsupervised on frames, then **frozen**; the M model consumes
  only its 32-d `mu` (the decoder is never used downstream).
- The M model is an LSTM with a **mixture density network** head (K=5 Gaussians per
  latent dimension), trained by negative log-likelihood on `z_t, a_t -> z_{t+1}`.
- Latent rollouts are sampled with a **temperature** parameter that scales the
  mixture weights and Gaussian variances, giving the paper's uncertainty control.
- The dataset uses short (100-step) rollouts after skipping the first 50 steps,
  because the camera zooms in during that period and the frames are unusable.

## Layout

```
envs/       CarRacing wrapper, random rollouts, dataset, preview scripts
vmodel/     VAE definition, ELBO loss, training, reconstruction, generation
mmodel/     MDN-RNN definition, mixture loss, sampling, smoke test
data/       (generated) rollouts and checkpoints — gitignored
```

## Setup

Python 3.10+ with `torch`, `torchvision`, `gymnasium[box2d]`, `numpy`, `pillow`,
`matplotlib`.

```bash
pip install torch torchvision gymnasium[box2d] numpy pillow matplotlib
```

## Usage

Run from the repository root so the package imports resolve.

```bash
# 1. Collect 500 rollouts of 100 frames each -> data/rollouts/rollout_XXXX.npz
python -m envs.collect_rollouts

# 2. Sanity-check the env / preview raw frames
python -m envs.show_frames

# 3. Train the V model -> vae_carracing.pth
python -m vmodel.train_vae

# 4. Inspect reconstructions (original vs. VAE)
python -m vmodel.reconstruct_frames

# 5. Shape/loss check for the M model
python -m mmodel.train_mdn
```

`vmodel/generate_vae.py` renders a single image (`ship7.png`) through the VAE; it
expects a checkpoint named `vae.pth`, so copy or rename `vae_carracing.pth` first.

## Status / limitations

- `mmodel/train_mdn.py` is a forward/backward smoke test on random tensors, not
  real training on encoded rollouts.
- No agent controller (`C`), no CMA-ES, no dream rollouts, and no reward-driven
  evaluation yet.
- There is no closed-loop "play" script and no comparison against the paper's
  reported scores.

## Reference

David Ha, Jurgen Schmidhuber (2018). *World Models*. arXiv:1803.10122.
