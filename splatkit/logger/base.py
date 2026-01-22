import logging
import sys
from typing import Literal

class SplatLogger:
    """
    Logger wrapper for splatkit using Python's built-in logging.
    
    Provides structured logging with module names for better debugging.
    
    Example:
        logger = SplatLogger(level="INFO")
        logger.info("Starting training", module="SplatTrainer")
        # Output: INFO:SplatTrainer:Starting training
    """
    
    def __init__(
        self,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        format: str = "%(levelname)s:SplatKit:%(module_name)s:%(message)s",
        log_file: str | None = None,
    ):
        """
        Initialize logger.
        
        Args:
            level: Logging level
            format: Log format string (default: LEVEL:MODULENAME:message)
            log_file: Optional file to write logs to
        """
        self._logger = logging.getLogger("splatkit")
        self._logger.setLevel(getattr(logging, level))
        self._logger.handlers.clear()  # Clear any existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(StructuredFormatter(format))
        self._logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level))
            file_handler.setFormatter(StructuredFormatter(format))
            self._logger.addHandler(file_handler)
    
    def debug(self, msg: str, module: str = "splatkit", **kwargs):
        """Log debug message."""
        self._logger.debug(msg, extra={"module_name": module, **kwargs})
    
    def info(self, msg: str, module: str = "splatkit", **kwargs):
        """Log info message."""
        self._logger.info(msg, extra={"module_name": module, **kwargs})
    
    def warning(self, msg: str, module: str = "splatkit", **kwargs):
        """Log warning message."""
        self._logger.warning(msg, extra={"module_name": module, **kwargs})
    
    def error(self, msg: str, module: str = "splatkit", **kwargs):
        """Log error message."""
        self._logger.error(msg, extra={"module_name": module, **kwargs})
    
    def critical(self, msg: str, module: str = "splatkit", **kwargs):
        """Log critical message."""
        self._logger.critical(msg, extra={"module_name": module, **kwargs})


class StructuredFormatter(logging.Formatter):
    """Custom formatter that handles module_name in extra dict."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add module_name to record if not present
        if not hasattr(record, "module_name"):
            record.module_name = "splatkit"
        
        return super().format(record)

