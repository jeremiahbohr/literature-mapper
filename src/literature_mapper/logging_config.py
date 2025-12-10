"""
Logging configuration for Literature Mapper.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    verbose: bool = False
) -> None:
    """
    Set up logging for Literature Mapper.
    
    Args:
        log_level: Minimum log level to capture
        log_file: Optional file path for log output
        verbose: Enable verbose logging (DEBUG level)
    """
    
    # Set level
    if verbose:
        log_level = "DEBUG"
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Console formatter with colors if supported
    if _supports_color():
        console_formatter = ColoredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Always detailed in files
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure third-party loggers
    _configure_third_party_loggers(numeric_level)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info("Logging configured - Level: %s, File: %s", log_level, log_file)

def _supports_color() -> bool:
    """Check if terminal supports colors."""
    import os
    return (
        hasattr(sys.stderr, "isatty") and sys.stderr.isatty() and
        os.environ.get('TERM') != 'dumb'
    )

def _configure_third_party_loggers(level: int) -> None:
    """Configure logging levels for third-party libraries."""
    third_party_loggers = [
        'urllib3',
        'requests', 
        'google',
        'google.auth',
        'google.generativeai',
        'sqlalchemy.engine',
        'pypdf'
    ]
    
    # Set to WARNING unless in DEBUG mode
    third_party_level = level if level <= logging.DEBUG else logging.WARNING
    
    for logger_name in third_party_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(third_party_level)

class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[95m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = self.formatTime(record, '%H:%M:%S')
        
        # Format message
        message = record.getMessage()
        
        # Add exception info if present
        if record.exc_info:
            message += '\n' + self.formatException(record.exc_info)
        
        # Create colored output
        return f"{timestamp} {color}[{record.levelname:8}]{reset} {record.name}: {message}"

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def setup_default_logging(corpus_path: Optional[Path] = None, verbose: bool = False) -> None:
    """
    Set up default logging configuration.
    
    Args:
        corpus_path: Optional corpus path for log file location
        verbose: Enable verbose logging
    """
    # Determine log file location
    log_file = None
    if corpus_path:
        log_file = corpus_path / "literature_mapper.log"
    else:
        # Try current directory, fall back to temp
        try:
            log_file = Path("literature_mapper.log")
            log_file.touch()  # Test write access
        except (PermissionError, OSError):
            import tempfile
            log_file = Path(tempfile.gettempdir()) / "literature_mapper.log"
    
    setup_logging(
        log_level="DEBUG" if verbose else "INFO",
        log_file=log_file,
        verbose=verbose
    )

# Export main functions
__all__ = [
    'setup_logging',
    'setup_default_logging',
    'get_logger'
]