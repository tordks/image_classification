import torchvision


class MNIST(torchvision.datasets.MNIST):
    """
    Wrap batch into a dictionary
    """

    def __getitem__(self, index: int):
        feature_key = "feature"
        label_key = "label"
        batch = {
            key: value
            for key, value in zip(
                (feature_key, label_key), super().__getitem__(index)
            )
        }
        return batch