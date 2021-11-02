from torch.utils.data import Dataset
from mnist_aug.mnist_augmenter import MNISTAug, DataManager


class MNISTAugDataset(Dataset):
    """
    A PyTorch Dataset for MNIST Detection and Captioning dataset.
    """

    def __init__(
        self,
        n_out: int,
        test_mode: bool = False,
        aug: MNISTAug = None,
        noisy: bool = False,
        get_class_captions: bool = False,
        get_relationships: bool = False,
        get_positional_labels: bool = False,
        get_positional_relationships: bool = False,
        get_relationship_captions: bool = False,
    ):
        """
        All params are same as params of aug.get_augmented()

        Every data points is a tuple of
        - image of shape (1, 112, 112)
        - List[Dict[str, Any]] of the format:
            {
                'class': int,
                'class_one_hot': List[int],
                'x1': int,
                'y1': int,
                'x2': int,
                'y2': int,
                'cx': float,
                'cy': float,
                'height': int,
                'width': int,
            }

        A default collate_fn will fail because there are uneven number of bboxes per image.
        """
        self.data_manager = DataManager()
        self.data_manager.load()

        self.aug = aug if aug is not None else MNISTAug()

        self.n_out = n_out
        self.noisy = noisy
        self.get_class_captions = get_class_captions
        self.get_relationships = get_relationships
        self.get_positional_labels = get_positional_labels
        self.get_positional_relationships = get_positional_relationships
        self.get_relationship_captions = get_relationship_captions

        self.x, self.y = (
            (self.data_manager.x_test, self.data_manager.y_test)
            if test_mode
            else (self.data_manager.x_train, self.data_manager.y_train)
        )

    def __len__(self):
        return self.n_out

    def __getitem__(self, idx):
        return self.aug.get_augmented(
            self.x,
            self.y,
            n_out=1,
            noisy=self.noisy,
            get_class_captions=self.get_class_captions,
            get_relationships=self.get_relationships,
            get_positional_labels=self.get_positional_labels,
            get_positional_relationships=self.get_positional_relationships,
            get_relationship_captions=self.get_relationship_captions,
        )


if __name__ == "__main__":
    dataset = MNISTAugDataset(n_out=10)
    x, y = next(iter(dataset))
    print(x.shape)
    print(y[0])
