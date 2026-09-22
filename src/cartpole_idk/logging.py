import logging


def configure_logging(log_level: int = logging.INFO) -> None:
    """Show CLI progress on stderr without interfering with stdout results.

    Args:
        log_level: Logging severity applied to the root logger.
    """
    msg_format = (
        "[%(asctime)s] p%(process)s {%(filename)s "
        "%(funcName)s:%(lineno)d} %(levelname)s - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=log_level,
        format=msg_format,
        datefmt=date_format,
    )
