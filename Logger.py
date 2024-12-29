import os

class Logger:
    @staticmethod
    def setup_logging(log_dir="NLP_FINAL_PROJECT/logs", log_file="training.log"):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def log(message):
        logging.info(message)
