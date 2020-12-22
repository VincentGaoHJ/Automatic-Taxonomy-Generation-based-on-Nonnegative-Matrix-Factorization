import logging
from utils.config import LOG_LEVEL

logger = logging.getLogger("TAROT LOG")
level = logging.getLevelName(LOG_LEVEL)
logger.setLevel(level)
fmt = "INVESTMENT LOG: %(asctime)s [%(levelname)s] %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(format=fmt, datefmt=date_fmt)

if __name__ == '__main__':
    logger.info("Log configured successfully!")
