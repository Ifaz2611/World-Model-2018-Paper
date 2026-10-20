from vmodel.vae import VAE
from vmodel.train_vae import CHECKPOINT
from envs.rollouts import RolloutFrames
import torch
import matplotlib.pyplot as plt

NUM_FRAMES = 8


def main():
    dset = RolloutFrames(limit=5)

    # Frames are pulled from across the rollouts so the row isn't eight views of the
    # same corner of one track.
    idx = torch.linspace(0, len(dset) - 1, NUM_FRAMES).long()
    X = torch.stack([dset[i] for i in idx])

    vae = VAE()
    vae.load_state_dict(torch.load(CHECKPOINT, weights_only=True))

    vae.eval()
    with torch.no_grad():
        X_recon, _, mu, logvar = vae(X)

    print(f"mu {tuple(mu.shape)}  mean |mu| {mu.abs().mean():.3f}  mean sigma {torch.exp(0.5 * logvar).mean():.3f}")

    fig, axes = plt.subplots(2, NUM_FRAMES, figsize=(1.6 * NUM_FRAMES, 3.6))
    for i in range(NUM_FRAMES):
        axes[0][i].imshow(X[i].permute(1, 2, 0).numpy())
        axes[1][i].imshow(X_recon[i].permute(1, 2, 0).numpy())
        axes[0][i].axis("off")
        axes[1][i].axis("off")

    axes[0][0].set_title("frame", loc="left")
    axes[1][0].set_title("reconstruction", loc="left")

    plt.tight_layout()
    plt.show()

if (__name__ == "__main__"):
    main()
