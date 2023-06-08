import logging


class LoggerPipe:
    """Of this logger there can only be one instance (write mode)
    """
    def __init__(self, path_to_log: str):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(path_to_log)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def write(self, message: str):
        self.logger.info(message)
