import logging

class Logger:
    @staticmethod
    def setup_logging(log_file="training.log"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def log(message):
        logging.info(message)
