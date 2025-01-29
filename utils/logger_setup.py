import logging

def setup_logging(log_file="pipeline.log", log_level=logging.DEBUG):
    """
    Configures the logging settings for the pipeline.
    Logs both to the console and to a file.

    Parameters:
    - log_file (str): Path to the log file.
    - log_level (int): Logging level, e.g., logging.DEBUG, logging.INFO.
    """
    # Define the logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Set up logging handlers
    handlers = [
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(log_file, mode="w")  # Log to a file
    ]

    # Configure the logging system
    logging.basicConfig(
        level=log_level,  # Set logging level
        format=log_format,  # Log format
        handlers=handlers  # Add handlers
    )

    # Add a success message after logging is configured
    logger = logging.getLogger(__name__)
    logger.info("✅ Logging is configured successfully.")
