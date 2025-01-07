import logging


def set_handler_level(console_handler, level=logging.INFO):
    console_handler.setLevel(level)


def set_handler_record_format(console_handler, extended_format: bool = False):
    if extended_format:
        formatter = logging.Formatter(
            "\x1b[31m%(asctime)s\x1b[0m | "
            "\x1b[32m%(filename)s:%(lineno)d\x1b[0m | "
            "\x1b[34m%(funcName)s\x1b[0m | "
            "\x1b[35m%(levelname)s\x1b[0m | "
            "%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "\x1b[31m%(asctime)s\x1b[0m | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    console_handler.setFormatter(formatter)


def set_logger_level_and_format(
    logger, level=logging.INFO, extended_format: bool = False
):
    logger.setLevel(level)
    for console_handler in logger.handlers:
        console_handler.setLevel(level)
        set_handler_record_format(console_handler, extended_format)


def set_all_loggers_level_and_format(level=logging.INFO, extended_format: bool = False):
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        set_logger_level_and_format(logger, level, extended_format)


def create_logger(name, level=logging.INFO, extended_format: bool = False):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    set_handler_record_format(console_handler, extended_format)
    logger.addHandler(console_handler)
    return logger
