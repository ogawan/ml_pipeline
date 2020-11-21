import logging

from regression_models.config import config
from regression_models.config import logging_config

VERSION_PATH = config.PACKAGE_ROOT / 'VERSION'
print("VERSION_PATH:", VERSION_PATH)

# Configure logger for use in package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False

with open(VERSION_PATH, 'r') as version_file:
    __version__ = version_file.read().strip()
    print("VERSION:", __version__)
