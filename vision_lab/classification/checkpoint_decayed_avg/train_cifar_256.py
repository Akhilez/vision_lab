import json
import re
from copy import deepcopy
from os import listdir, makedirs
from os.path import isdir, join, dirname

import pandas as pd
import torch
from torch import nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm


BASE_DIR = dirname(dirname(dirname(__file__)))
PARENT_PATH = dirname(__file__)


def split_train_test():
    data_path = "/Users/akhildevarashetti/code/vision_lab/data/256_ObjectCategories"
    dirs = sorted(listdir(data_path))
    dirs = [d for d in dirs if isdir(join(data_path, d))]

    class_id, class_name, class_dir = [], [], []
    pattern = r"^([0-9]{3})\.(.*)"
    for d in dirs:
        if re.match(pattern, d):
            match = re.match(pattern, d)
            class_id.append(int(match.group(1)))
            class_name.append(match.group(2))
            class_dir.append(d)

    # For each class, split the images into train and test sets
    train_frac = 0.8
    train_images, test_images = {}, {}
    for class_dir in tqdm(class_dir):
        image_names = listdir(join(data_path, class_dir))
        image_names = [i for i in image_names if i.endswith(".jpg")]
        num_images = len(image_names)
        num_train = int(num_images * train_frac)
        # Randomly shuffle the images
        np.random.shuffle(image_names)
        train_images[class_dir] = image_names[:num_train]
        test_images[class_dir] = image_names[num_train:]

    # Save these splits to disk
    with open(join(data_path, "train_images.json"), "w") as f:
        json.dump(train_images, f, indent=4)
    with open(join(data_path, "test_images.json"), "w") as f:
        json.dump(test_images, f, indent=4)


class Cifar256ClassificationDataset(Dataset):
    def __init__(self, data_path, is_train=True, height=256, width=256):
        self.data_path = data_path
        self.is_train = is_train
        self.h, self.w = height, width
        self.images_json = json.load(open(join(data_path, f"{'train' if is_train else 'test'}_images.json")))
        self.classes = sorted(list(self.images_json.keys()))
        self.num_classes = len(self.classes)
        print(f"Number of classes: {self.num_classes}")
        self.image_paths = []
        self.labels = []
        for class_name, image_names in self.images_json.items():
            for image_name in image_names:
                self.image_paths.append(join(data_path, class_name, image_name))
                self.labels.append(self.classes.index(class_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.w, self.h))
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image)

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def create_model():
    # model = nn.Sequential(
    #     nn.Conv2d(3, 128, 3, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.BatchNorm2d(128),
    #     nn.Conv2d(128, 128, 3, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.BatchNorm2d(128),
    #     nn.Conv2d(128, 256, 3, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.BatchNorm2d(256),
    #     nn.Conv2d(256, 256, 3, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.BatchNorm2d(256),
    #     nn.Conv2d(256, 512, 3, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.BatchNorm2d(512),
    #     nn.Conv2d(512, 512, 3, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.BatchNorm2d(512),
    #     nn.Conv2d(512, 1024, 3, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.BatchNorm2d(1024),
    #     nn.Flatten(),
    #     nn.Linear(1024 * 2 * 2, 257),
    # )
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model.classifier = nn.Linear(1280, 257)
    return model


def eval_checkpoint(checkpoint):
    model = create_model()

    batch_size = 128
    device = "cuda:1"
    dataloader_workers = 4
    data_path = f"{BASE_DIR}/data/256_ObjectCategories"

    # Set checkpoint
    model.load_state_dict(checkpoint)
    model = model.to(device)

    test_dataset = Cifar256ClassificationDataset(data_path, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

    model.eval()
    with torch.inference_mode():
        total, correct = 0, 0
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return correct / total


def eval_weight_aggregation():
    checkpoints = listdir(join(PARENT_PATH, "checkpoints"))
    checkpoints = [c for c in checkpoints if c.endswith(".pth")]
    pattern = r"^model_(\d+).pth"
    checkpoint_ids = []
    for c in checkpoints:
        match = re.match(pattern, c)
        checkpoint_ids.append(int(match.group(1)))
    checkpoint_ids = sorted(checkpoint_ids, reverse=True)
    # max_id = 15
    # checkpoint_ids = [c for c in checkpoint_ids if c <= max_id]
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Checkpoint IDs: {checkpoint_ids}")

    checkpoints = {
        checkpoint_id: torch.load(join(PARENT_PATH, "checkpoints", f"model_{checkpoint_id}.pth"), map_location="cpu")
        for checkpoint_id in checkpoint_ids
    }

    data = {
        "epoch": [],
        "decay_rate": [],
        "val_accuracy": [],
    }
    for epoch in checkpoint_ids:
        accuracy = eval_checkpoint(checkpoints[epoch])
        data["epoch"].append(epoch)
        data["decay_rate"].append(0)
        data["val_accuracy"].append(accuracy)

        with open(join(PARENT_PATH, "results.json"), "w") as f:
            json.dump(data, f, indent=4)

    decay_rates = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    for decay_rate in decay_rates:
        for max_epoch in range(1, len(checkpoint_ids)):
            checkpoint_ids_subset = [c for c in checkpoint_ids if c <= max_epoch]
            print(f"Decay rate: {decay_rate}, Max epoch: {max_epoch}, {checkpoint_ids_subset=}")

            weights = decay_rate ** np.arange(len(checkpoint_ids_subset))
            weights /= np.sum(weights)
            print(f"Weights: {weights}")

            checkpoint_agg = None
            for i, checkpoint_id in enumerate(checkpoint_ids_subset):
                checkpoint = deepcopy(checkpoints[checkpoint_id])
                if checkpoint_agg is None:
                    checkpoint_agg = checkpoint
                    for k in checkpoint_agg.keys():
                        if "num_batches_tracked" in k:
                            continue
                        checkpoint_agg[k] *= weights[i]
                else:
                    for k in checkpoint_agg.keys():
                        if "num_batches_tracked" in k:
                            continue
                        checkpoint_agg[k] += checkpoint[k] * weights[i]

            accuracy_agg = eval_checkpoint(checkpoint_agg)
            data["epoch"].append(max_epoch)
            data["decay_rate"].append(decay_rate)
            data["val_accuracy"].append(accuracy_agg)
            print(accuracy_agg)

            with open(join(PARENT_PATH, "results.json"), "w") as f:
                json.dump(data, f, indent=4)



def plot_results():
    df = pd.read_json(join(PARENT_PATH, "results.json"))



def train():
    model = create_model()

    sample_in = torch.randn(1, 3, 256, 256)
    sample_out = model(sample_in)
    print(sample_out.shape)

    summary(model, (3, 256, 256), device="cpu")

    n_epochs = 25
    batch_size = 128
    learning_rate = 1e-3
    dataloader_workers = 4
    device = "cuda:0"
    data_path = f"{BASE_DIR}/data/256_ObjectCategories"
    checkpoints_path = f"{BASE_DIR}/classification/checkpoint_decayed_avg/checkpoints"
    makedirs(checkpoints_path, exist_ok=True)

    model = model.to(device)
    # torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_dataset = Cifar256ClassificationDataset(data_path, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)

    test_dataset = Cifar256ClassificationDataset(data_path, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # if i % 10 == 0:
                # print(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")
            pbar.set_description(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")

        model.eval()
        with torch.inference_mode():
            total, correct = 0, 0
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Epoch {epoch}, Test Accuracy: {correct / total}")

        torch.save(model.state_dict(), join(checkpoints_path, f"model_{epoch}.pth"))


if __name__ == "__main__":
    # train()
    eval_weight_aggregation()


"""
scp \
/Users/akhildevarashetti/code/vision_lab/data/256_ObjectCategories.tar.gz \
akhil@glados.acvlabs.acvauctions.com:/home/akhil/code/vision_lab/data

Epoch 0, Test Accuracy: 0.18531018964963034
Epoch 1, Test Accuracy: 0.27033108325297334
Epoch 2, Test Accuracy: 0.34490517518482805
Epoch 3, Test Accuracy: 0.3831565413050466
Epoch 4, Test Accuracy: 0.39665702346512377
Epoch 5, Test Accuracy: 0.4225329476052716
Epoch 6, Test Accuracy: 0.42912246865959497
Epoch 7, Test Accuracy: 0.4323368691738991
Epoch 8, Test Accuracy: 0.43989071038251365
Epoch 9, Test Accuracy: 0.44776599164255865
Epoch 10, Test Accuracy: 0.44889103182256507
Epoch 11, Test Accuracy: 0.45065895210543233
Epoch 12, Test Accuracy: 0.4532304725168756
Epoch 13, Test Accuracy: 0.4525875924140148
Epoch 14, Test Accuracy: 0.4533911925425908
Epoch 15, Test Accuracy: 0.453551912568306
Epoch 16, Test Accuracy: 0.45596271295403407
Epoch 17, Test Accuracy: 0.4562841530054645
Epoch 18, Test Accuracy: 0.45580199292831886
Epoch 19, Test Accuracy: 0.4572484731597557
Epoch 20, Test Accuracy: 0.45949855351976854
Epoch 21, Test Accuracy: 0.4574091931854709
Epoch 22, Test Accuracy: 0.4572484731597557
Epoch 23, Test Accuracy: 0.1647380263580842
Epoch 24, Test Accuracy: 0.36853101896496304
"""
