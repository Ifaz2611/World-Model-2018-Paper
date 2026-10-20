from vmodel.vae import VAE, elbo_loss
from envs.rollouts import RolloutFrames
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
CHECKPOINT = "vae_carracing.pth"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"training on {device}")

    # train_dset = torchvision.datasets.CIFAR10(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=transforms.Compose([
    #         transforms.Resize((64, 64)),
    #         transforms.ToTensor()
    #     ])
    # )

    train_dset = RolloutFrames()
    print(f"{len(train_dset)} frames")

    # The frames are already decoded and resident in memory, so worker processes would
    # only copy the array around.
    data_loader = DataLoader(
        train_dset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    vae = VAE().to(device)
    optimizer = optim.Adam(params=vae.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for step, image in enumerate(data_loader):
            image = image.to(device)

            optimizer.zero_grad()

            X_recon, _, mu, logvar = vae(image)

            loss = elbo_loss(X_recon, image, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 100 == 0:
                print(f"epoch {epoch + 1}/{EPOCHS}  step {step}/{len(data_loader)}  loss {loss.item():.3f}")

        print(f"epoch {epoch + 1}/{EPOCHS} done  mean loss {running_loss / len(data_loader):.3f}")

        # The V model is trained on its own and then frozen, so the checkpoint is written
        # every epoch and the M model just loads it.
        torch.save(vae.state_dict(), CHECKPOINT)

    torch.save(vae.state_dict(), CHECKPOINT)

if (__name__ == "__main__"):
    main()
