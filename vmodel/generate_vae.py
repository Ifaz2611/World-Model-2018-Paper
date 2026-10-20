import torch
import torchvision
from torchvision.transforms import v2
from vmodel.vae import VAE
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = Image.open("ship7.png")

    tensor = v2.functional.to_image(img)
    tensor = v2.functional.resize(tensor, (64, 64))
    tensor = tensor.float() / 255.0
    tensor = tensor.unsqueeze(0)  # (3, 64, 64) -> (1, 3, 64, 64)

    print(tensor.shape)

    vae = VAE()
    vae.load_state_dict(torch.load("vae.pth", weights_only=True))

    vae.eval()
    with torch.no_grad():
        gen_image, _, _, _ = vae(tensor)

    image_res = gen_image[0].permute(1, 2, 0).numpy()

    plt.imshow(image_res)
    plt.show()
    

if (__name__ == "__main__"):
    main()