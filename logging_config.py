import sys
import logging
from pathlib import Path

try:
    from google.colab import files

    COLAB = True
except ModuleNotFoundError:
    COLAB = False

# create logger
logger = logging.getLogger('E2E')
logger.setLevel(logging.DEBUG)

if COLAB:
    # Remove the root logger's handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

log_file = f"outputs/output.txt"
Path(log_file).parent.mkdir(parents=True, exist_ok=True)
with open(log_file, 'w') as f:
    f.write('')
fh1 = logging.FileHandler(log_file)
fh1.setLevel(logging.DEBUG)

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
