import sys
import logging
import time
from pathlib import Path

# create logger
logger = logging.getLogger('E2E')
logger.setLevel(logging.DEBUG)

log_file = f"outputs/output.log"
Path(log_file).parent.mkdir(parents=True, exist_ok=True)
with open(log_file, 'w') as f:
    f.write('')
fh1 = logging.FileHandler(log_file)
fh1.setLevel(logging.DEBUG)

# fh2 = logging.FileHandler('spam.log')
# fh2.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
# formatter = logging.Formatter("\n# %(asctime)s [%(name)-12s] [%(levelname)-5.5s]  \n%(message)s")
# formatter = logging.Formatter("[%(name)-20.20s] [%(levelname)-1.1s]  %(message)s")
formatter = logging.Formatter("%(message)s")
fh1.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh1)
logger.addHandler(ch)

logger.info("Logger Started!")


def get_logger(name):
    return logging.getLogger(f'E2E.{name}')
