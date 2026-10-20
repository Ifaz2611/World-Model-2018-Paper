from mmodel.mdnmodel import MModel, mdn_loss
import torch

def main():
    mmodel = MModel()

    test_tensor = torch.randn(size=(1, 20, 32))

    pi_logits, mu, logvar, hidden = mmodel(test_tensor)

    print(f"pi_logits {tuple(pi_logits.shape)}  mu {tuple(mu.shape)}  logvar {tuple(logvar.shape)}")

    # The model predicts z_{t+1} from z_t, so the target is the sequence
    # shifted by one step.
    loss = mdn_loss(pi_logits[:, :-1], mu[:, :-1], logvar[:, :-1], test_tensor[:, 1:])
    print(f"loss {loss.item():.3f}")

    sample = mmodel.sample(pi_logits, mu, logvar, temperature=1.0)
    print(f"sample {tuple(sample.shape)}")

if (__name__ == "__main__"):
    main()
