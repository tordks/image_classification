from enum import Enum

from loguru import logger
from rich.logging import RichHandler


# TODO: make logger work well with tqdm/PL progress bar
logger.remove()
logger.add(
    sink=RichHandler(show_time=True, log_time_format="%Y-%m-%d %H:%M:%S"),
    format="{message}",
)


# TODO: New name? might be confusing with the LightningModule setup argument
#       being called stage
class Stage(Enum):
    training_step = "training_step"
    validation_step = "validation_step"
    training_epoch_end = "training_epoch_end"
    validation_epoch_end = "validation_epoch_end"
    on_train_epoch_end = "on_train_epoch_end"
    on_validation_epoch_end = "on_validation_epoch_end"
