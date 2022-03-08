import abc
from pathlib import Path
import shutil

# from fastcore.meta import delegates
from matplotlib.figure import Figure
from neptune.new.types import File
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger


# NOTE: Having these custom loggers might be a footgun. It could reduce the
# flexibility later, but for now it is nice to be able to quickly experiment and
# shift between the different loggers. In the future we might converge to one
# logger.

# TODO: make abstract class that implements the interface?
# TODO: make use of log_hyperparams()?


class CustomLoggerInterface(metaclass=abc.ABCMeta):
    """
    Interface that each logger must adhere to defining new loggers.
    """

    @abc.abstractmethod
    def log_metadata(self, key: str, metadata):
        raise NotImplementedError

    @abc.abstractmethod
    def log_image(self, key: str, image: Figure, step: int):
        raise NotImplementedError

    @abc.abstractmethod
    def save_file(self, key: str, fpath: Path):
        raise NotImplementedError


# @delegates()
class CustomNeptuneLogger(NeptuneLogger):
    """
    NeptuneLogger that adds certain functions to adhere to a common standard
    logging interface.
    """

    # TODO: Create project if it does not exist (already handled by PL)?

    def log_metadata(self, key: str, metadata):
        """
        Log metadata
        """
        self.experiment[key].log(metadata)

    def log_image(self, key: str, image: Figure, step: int):
        self.experiment[key].log(File.as_image(image), step=step)

    def save_file(self, key: str, fpath: Path):
        """
        Uploads an arbitrary file artifact. Eg. a model or config.
        """
        self.experiment[key].upload(str(fpath))


# @delegates()
class CustomTensorBoardLogger(TensorBoardLogger):
    """
    TensorBoardLogger that adds certain functions to adhere to a common standard
    logging interface.
    """

    def log_metadata(self, key: str, metadata: dict[str]):
        """
        Log metadata. For TensorboardLogger the metadata is already contained
        locally, hence just pass.
        """
        pass

    def log_image(self, key: str, image: Figure, step: int):
        self.experiment.add_figure(key, image, step)

    def save_file(self, key: str, fpath: Path):
        """
        copies a file into the logging folder. Eg. a model or config.
        """
        log_dir = Path(self.log_dir)
        if not log_dir.exists():
            log_dir.mkdir()

        shutil.move(fpath, log_dir / fpath.name)
