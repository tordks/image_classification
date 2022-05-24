from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, sample_mapping: dict[int, str]):
        """
        Wrap a dataset so that it outputs samples as a dictionary instead of a
        tuple/list.

        :param dataset: dataset to wrap
        :param sample_mapping: mapping from each idx in the dataset sample to a
                               key
        """
        self.dataset = dataset
        self.sample_mapping = sample_mapping

    def __getitem__(self, sample_idx: int):
        return {
            self.sample_mapping[i]: element
            for i, element in enumerate(self.dataset[sample_idx])
        }

    def __len__(self):
        return len(self.dataset)
