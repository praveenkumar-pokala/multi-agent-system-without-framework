"""
Configurable logging setup for the multi-agent system.

This module configures the Loguru logger to output logs to
standard output and a rotating file in the ``logs``
directory. If the directory does not exist it will be
created on import. Logs include timestamps and log levels
for easy debugging. The file log rotates at 1 MB and
retains up to 10 backups.
"""
import os
import sys
from loguru import logger

# Ensure the logs directory exists
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Remove the default handler and add our own
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time}</green> <level>{message}</level>")
logger.add(
    os.path.join(LOG_DIR, "multi_agent_system.log"),
    rotation="1 MB",
    retention="10 days",
    level="DEBUG",
    format="{time} {level} {message}",
)