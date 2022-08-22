import logging


def enable_logger():
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s: %(message)s', datefmt='%H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger('origami')
    logger.addHandler(stream_handler)
