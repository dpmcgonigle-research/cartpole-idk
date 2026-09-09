import logging


def configure_logging() -> None:
    """Show CLI progress on stderr without interfering with stdout results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
