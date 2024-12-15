import logging
import os


class FilenameFormatter(logging.Formatter):
    def format(self, record):
        record.filename = os.path.basename(record.pathname)
        return super().format(record)


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler and set level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter
    # formatter = FilenameFormatter(
    #    "%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S'
    # )
    # formatter = FilenameFormatter("%(message)s")

    formatter = FilenameFormatter(
        "\x1b[31;20m%(asctime)s\x1b[0m %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    return logger
