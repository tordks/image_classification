from typing import Union
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_grayscale


class ColorisationWrapper(Dataset):
    def __init__(
        self, dataset: Dataset, sample_mapping: dict[Union[int, str], str]
    ):
        """
        Wrap an image dataset so that it outputs the image as target and a
        grayscale as features. Also wrap samples in a dictionary instead of a
        tuple/list to adhere to the convention of "feature" and "target" keys

        :param dataset: dataset to wrap
        :param sample_mapping: mapping from each idx in the dataset sample to a
                               key
        """
        self.dataset = dataset
        self.sample_mapping = sample_mapping

    def __getitem__(self, sample_idx: int):
        sample = {
            self.sample_mapping[i]: element
            for i, element in enumerate(self.dataset[sample_idx])
        }
        sample["target"] = sample["feature"]
        sample["feature"] = to_grayscale(sample["feature"])
        return sample

    def __len__(self):
        return len(self.dataset)
