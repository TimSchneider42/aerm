def _setup_logging():
    import logging
    import sys

    logger = logging.getLogger("main")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(lambda r: r.levelno <= logging.WARNING)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    logger.addHandler(stderr_handler)
    logger.setLevel(logging.INFO)
    return logger


logger = _setup_logging()
