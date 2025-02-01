from os import listdir
from os.path import abspath, dirname

import numpy as np
import torch
import cv2
import imageio

from vision_lab.generation.diffusion.mnist_diffusion.train_diffusion import SqueezeDiffusionModel
from vision_lab.generation.diffusion.mnist_diffusion.unet import UNet
from vision_lab.generation.diffusion.mnist_diffusion.unet2 import Unet


class DiffusionProcessor:
    def __init__(self):
        self.T = 100
        self.b = np.linspace(0.001, 0.1, self.T)
        self.a = 1 - self.b
        self.a_bar = np.cumprod(self.a)

        self.a_sqrt = np.sqrt(self.a)
        self.b_sqrt = np.sqrt(self.b)
        self.a_bar_sqrt = np.sqrt(self.a_bar)
        self.one_minus_a_bar_sqrt = np.sqrt(1 - self.a_bar)

    def get_inputs_and_outputs(self, image: np.ndarray, t: int | float):
        if type(t) is float:
            assert 0 <= t <= 1
            t = int(t * self.T)
            assert 0 <= t < self.T

        noise = np.random.randn(*image.shape).astype(np.float32)
        image_t = self.a_bar_sqrt[t] * image + self.one_minus_a_bar_sqrt[t] * noise

        return image_t, noise

    def backward_akhil(self, image: torch.tensor, noise: torch.tensor, t: int | float):
        if type(t) is float:
            assert 0 <= t <= 1
            t = int(t * self.T)
            assert 0 <= t < self.T

        image_prev = (image - (self.b_sqrt[t] * noise)) / self.a_sqrt[t]

        return image_prev

    def backward_ddpm1(self, image: torch.tensor, noise: torch.tensor, t: int | float):
        if type(t) is float:
            assert 0 <= t <= 1
            t = int(t * self.T)
            assert 0 <= t < self.T

        # image_prev = (image - (self.b_sqrt[t] * noise)) / self.a_sqrt[t]

        mean = (image - (self.b[t] / self.one_minus_a_bar_sqrt[t] * noise)) / self.a_sqrt[t]
        if t == 0:
            return mean
        else:
            variance = (1 - self.a_bar[t-1]) / (1 - self.a_bar[t])
            variance = variance * self.b[t]
            sigma = np.sqrt(variance)

            z = torch.randn(*image.shape).to(image.device)
            image_prev = mean + (sigma * z)

            return image_prev


def save_animation(path, images):
    # images: (n, 1, H, W)

    images = [image.detach().cpu().numpy() for image in images]
    images = [np.transpose(image, (1, 2, 0)) for image in images]
    images = [image - image.min() for image in images]
    images = [image / image.max() for image in images]
    images = [(image * 255).astype(np.uint8) for image in images]
    images = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in images]

    imageio.mimsave(path, images)


def infer():
    PARENT_DIR = dirname(abspath(__file__))
    checkpoint_dir = f"{PARENT_DIR}/checkpoints"
    checkpoint_name = sorted(listdir(checkpoint_dir), reverse=True)[0]
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_name}"

    checkpoint_path = f"{PARENT_DIR}/checkpoints/model_0020.pth"

    device = "cuda:3" if torch.cuda.is_available() else "mps"
    model = SqueezeDiffusionModel(in_channels=2, w=32, depth=8, out_channels=1).to(device)
    # model = UNet(in_channels=2, out_channels=1, depth=4, initial_filters=32).to(device)
    # model = Unet({
    #     "im_channels": 1,
    #     "down_channels": [16, 32, 64, 128],
    #     "mid_channels": [128, 128, 64],
    #     "time_emb_dim": 128,
    #     "down_sample": [True, True, False],
    #     "num_down_layers": 2,
    #     "num_mid_layers": 2,
    #     "num_up_layers": 2,
    # }).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    diffuser = DiffusionProcessor()

    image = torch.randn(1, 28, 28).to(device)
    animation = [image]
    model.eval()
    with torch.no_grad():
        for t in torch.arange(diffuser.T - 1, -1, -1):
            # inputs = model.encode_t(image.unsqueeze(0), t.view(1, 1).to(image.device))

            inputs = image.unsqueeze(0)
            t = t.view(1, ).to(image.device)

            noise = model(inputs, t)
            # image = diffuser.backward_akhil(image, noise[0], int(t))
            image = diffuser.backward_ddpm1(image, noise[0], int(t))
            animation.append(image)

    # save_animation(f"{PARENT_DIR}/animations/animation-akhil.mp4", animation)
    save_animation(f"{PARENT_DIR}/animations/animation-ddpm.mp4", animation)


if __name__ == '__main__':
    infer()
