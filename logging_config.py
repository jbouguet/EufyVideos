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
    #     "\x1b[31m%(asctime)s\x1b[0m - "
    #     "\x1b[32m%(filename)s:%(lineno)d\x1b[0m - "
    #     "\x1b[34m%(funcName)s\x1b[0m - "
    #     "\x1b[35m%(levelname)s\x1b[0m - "
    #     "%(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    # Create formatter
    formatter = FilenameFormatter(
        "\x1b[31m%(asctime)s\x1b[0m - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    return logger
