from os.path import abspath, dirname, join

import torch
from torch import nn
import torch.nn.functional as F
import cv2
from moviepy import ImageSequenceClip
import wandb

PROJECT_DIR = dirname(abspath(__file__))


class MobileNetV2Features(nn.Module):
    layer_indices = [11, 9, 7, 5]

    def __init__(self):
        super().__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        """
        Has 18 sequential layers in model.features.
        We want to extract the features of layers [18, 16, 14, 12, 10] for example.
        """
        self.cnn_layers = nn.ModuleList(model.features[:max(self.layer_indices) + 1])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.cnn_layers):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features


class VGG19Features(nn.Module):
    layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]

    def __init__(self):
        super().__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        """
        Has 37 sequential layers in model.features.
        We want to extract the features of layers [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28] for example.
        """
        self.cnn_layers = nn.ModuleList(model.features[:max(self.layer_indices) + 1])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.cnn_layers):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features


def load_image(image_path, w=224, h=224):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (w, h))
    image = image / 255.0
    # Imagenet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std  # Shape: (224, 224, 3)
    image = image.transpose((2, 0, 1))  # Shape: (3, 224, 224)
    image = torch.tensor(image).float()
    return image


def denormalize_image(image):
    """
    image is a tensor of shape (3, 224, 224)
    It is normalized with mean and std of ImageNet.
    The output should be a numpy array of shape (224, 224, 3) in the range [0, 255].
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image.detach().cpu().numpy()
    image = image.transpose((1, 2, 0))
    image = image * std + mean
    image = image * 255.0
    image = image.clip(0, 255)
    return image


def save_video(frames, output_path, fps=30):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path)


def get_gram(tensor):
    b, c, h, w = tensor.size()
    
    features = tensor.view(b * c, h * w)

    gram = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    gram = gram / (b * c * h * w)  # Shape: (b*c, b*c)

    return gram


def train_target_features():
    target_image_path = join(PROJECT_DIR, "target_images", "target10.jpg")
    style_image_path = join(PROJECT_DIR, "style_images", "style11.png")
    num_steps = 2000
    size = 224
    device = "cuda:0" if torch.cuda.is_available() else "mps"
    save_every = 50
    learning_rate = 1e-1
    # lr_initial = 2e-2
    # lr_final = 1e-3
    # lr_decay_steps = 100
    # lr_decay = (lr_final / lr_initial) ** (1 / (num_steps // lr_decay_steps))

    wandb.login()
    wandb.init(
        project="style-transfer",
        mode="online",
        config={
            "model": "MobileNetV2",
            "layers": VGG19Features.layer_indices,
            "num_steps": num_steps,
            "resolution": size,
            # "lr_initial": lr_initial,
            # "lr_final": lr_final,
            # "lr_decay_steps": lr_decay_steps,
            # "lr_decay": lr_decay,
        },
    )

    model = VGG19Features().to(device)

    target_image = load_image(target_image_path, size, size).to(device)
    style_image = load_image(style_image_path, size, size).to(device)

    inputs = torch.stack([target_image, style_image])
    model.eval()
    with torch.no_grad():
        features = model(inputs)  # Shape: (n_layers, 2, c*, h*, w*)
    target_features, style_features = zip(*features)
    style_grams = [get_gram(style.unsqueeze(0)) for style in style_features]

    latent = torch.randn(1, 3, size, size, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([latent], lr=learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=lr_decay)

    model.train()
    animation_frames = [denormalize_image(latent[0])]
    for step in range(num_steps):
        latent_features = model(latent)  # Shape: (n_layers, 1, c*, h*, w*)

        # Content loss
        loss_content = 0.0
        for (latent_i,), target in zip(latent_features, target_features):
            loss_content += F.mse_loss(latent_i, target)

        # Style loss
        loss_style = 0.0
        for latent_i, style_gram in zip(latent_features, style_grams):
            latent_gram = get_gram(latent_i)
            loss_style += F.mse_loss(latent_gram, style_gram)

        loss = (0.001 * loss_content) + (100 * loss_style)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learning_rate = optimizer.param_groups[0]["lr"]
        # lr_scheduler.step()
        if step % save_every == 0:
            animation_frames.append(denormalize_image(latent[0]))
            print(f"Step {step}, Loss: {loss.item()}")
            wandb.log(
                {
                    "loss": loss.item(),
                    "learning_rate": learning_rate,
                    "step": step,
                    "loss_content": loss_content.item(),
                    "loss_style": loss_style.item(),
                }
            )

    animation_path = join(PROJECT_DIR, "output", "animation.mp4")
    save_video(animation_frames, animation_path, fps=30)
    wandb.log({"animation": wandb.Video(animation_path)})


def test_interface():
    image_path = join(PROJECT_DIR, "style_images", "style1.jpg")
    image = load_image(image_path)
    image = image.unsqueeze(0)
    print(f"Image shape: {image.shape}")

    model = VGG19Features()
    # x = torch.randn(1, 3, 224, 224)
    features = model(image)
    for i, feature in enumerate(features):
        print(f"Feature {i} shape: {feature.shape}")


if __name__ == "__main__":
    # test_interface()
    train_target_features()
