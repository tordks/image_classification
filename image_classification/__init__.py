from loguru import logger
from rich.logging import RichHandler


logger.remove()
logger.add(
    sink=RichHandler(show_time=True, log_time_format="%Y-%m-%d %H:%M:%S"),
    format="{message}",
)
