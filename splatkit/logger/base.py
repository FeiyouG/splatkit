import logging
import sys
from typing import Literal, Sequence
from ..modules import SplatRenderPayloadT
from ..modules.base import SplatBaseModule

class SplatLogger(
    SplatBaseModule[SplatRenderPayloadT],
):
    """
    Logger wrapper for splatkit using Python's built-in logging.    
    Provides structured logging with module names for better debugging.
    
    Example:
        >>> from splatkit.logger import SplatLogger
        >>> logger = SplatLogger(level="INFO")
        >>> logger.info("Starting training", module="SplatTrainer")
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
        
        self._world_rank = 0,
        self._world_size = 1
    
    def on_setup(
        self,
        logger: "SplatLogger",
        renderer: SplatBaseModule[SplatRenderPayloadT],
        data_provider: SplatBaseModule[SplatRenderPayloadT],
        loss_fn: SplatBaseModule[SplatRenderPayloadT],
        densification: SplatBaseModule[SplatRenderPayloadT],
        modules: Sequence[SplatBaseModule[SplatRenderPayloadT]], 
        max_steps: int,
        world_rank: int = 0,
        world_size: int = 1,
        scene_scale: float = 1.0,
    ):
        self._world_rank = world_rank
        self._world_size = world_size
        self.info(f"Successfully set up logger; will log messages only on rank 0", module=self.module_name)
    
    def debug(self, msg: str, module: str = "splatkit", **kwargs):
        """Log debug message only on rank 0."""
        if self._world_rank != 0:
            return
        
        self._logger.debug(msg, extra={"module_name": module, **kwargs})
    
    def info(self, msg: str, module: str = "splatkit", **kwargs):
        """Log info message only on rank 0."""
        if self._world_rank != 0:
            return
        
        self._logger.info(msg, extra={"module_name": module, **kwargs})
    
    def warning(self, msg: str, module: str = "splatkit", **kwargs):
        """Log warning message only on rank 0."""
        if self._world_rank != 0:
            return
        
        self._logger.warning(msg, extra={"module_name": module, **kwargs})
    
    def error(self, msg: str, module: str = "splatkit", **kwargs):
        """Log error message only on rank 0."""
        if self._world_rank != 0:
            return
        
        self._logger.error(msg, extra={"module_name": module, **kwargs})
    
    def critical(self, msg: str, module: str = "splatkit", **kwargs):
        """Log critical message only on rank 0."""
        if self._world_rank != 0:
            return
        
        self._logger.critical(msg, extra={"module_name": module, **kwargs})


class StructuredFormatter(logging.Formatter):
    """Custom formatter that handles module_name in extra dict."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add module_name to record if not present
        if not hasattr(record, "module_name"):
            record.module_name = "splatkit"
        
        return super().format(record)