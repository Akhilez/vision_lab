from os import makedirs
from os.path import join
import yaml
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm
from mnist_aug.mnist_augmenter import MnistAugDataManager, MNISTAug


def main():
    """
    Saves images and annotations.yaml file in the following format:
    <image_name>:
        height: 112
        width: 112
        image_id: 0
        annotations:
          - class: 6
            cx: 30.5
            cy: 76.5
            height: 45
            id: 2
            type: number
            width: 45
            x1: 8
            x2: 53
            y1: 54
            y2: 99
    :return:
    """

    cfg = DictConfig(
        {
            "output_dataset_path": "/Users/akhil/code/vision_lab/data/mnist_detection/sample/test",
            "size": 100,
            "train": False,
        }
    )

    makedirs(cfg.output_dataset_path, exist_ok=True)

    data_manager = MnistAugDataManager()
    data_manager.load(train=cfg.train)
    x = data_manager.x_train if cfg.train else data_manager.x_test
    y = data_manager.y_train if cfg.train else data_manager.y_test

    aug = MNISTAug()
    xs, ys = aug.get_augmented(
        x,
        y,
        n_out=cfg.size,
        # noisy=False,
        # get_class_captions=self.get_class_captions,
        # get_relationships=self.get_relationships,
        # get_positional_labels=self.get_positional_labels,
        # get_positional_relationships=self.get_positional_relationships,
        # get_relationship_captions=self.get_relationship_captions,
    )
    _, height, width = xs.shape

    output_dict = {}

    for i in tqdm(range(cfg.size)):
        file_name = f"image_{i}.png"
        file_path = join(cfg.output_dataset_path, file_name)
        for a in ys[i]:
            del a["class_one_hot"]

        output_dict[file_name] = {
            "height": height,
            "width": width,
            "image_id": i,
            "annotations": ys[i],
        }

        image = xs[i] * 255
        image = Image.fromarray(image).convert("L")
        image.save(file_path)

    with open(join(cfg.output_dataset_path, "annotations.yaml"), "w") as yaml_file:
        yaml.safe_dump(output_dict, yaml_file)

    print(f"Saved the dataset to {cfg.output_dataset_path}")


if __name__ == "__main__":
    main()
