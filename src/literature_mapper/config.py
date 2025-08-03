"""
Configuration management for Literature Mapper.

Provides simple configuration with environment variable support
and sensible defaults.
"""

import os
from dataclasses import dataclass
import logging
from .exceptions import ValidationError

logger = logging.getLogger(__name__)

# Version and model defaults
VERSION = "0.1.0"
DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-pro"

# Processing defaults
DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2

@dataclass
class LiteratureMapperConfig:
    """
    Configuration for Literature Mapper with environment variable support.
    """
    
    # Core settings
    api_key: str = None
    model_name: str = DEFAULT_MODEL
    
    # Processing limits
    max_file_size: int = DEFAULT_MAX_FILE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: int = DEFAULT_RETRY_DELAY
    
    # Logging
    log_level: str = "INFO"
    verbose: bool = False
    
    def __post_init__(self):
        """Load from environment and validate."""
        self._load_from_environment()
        self._validate_config()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # API configuration
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
        
        # Model selection
        env_model = os.getenv("LITERATURE_MAPPER_MODEL")
        if env_model:
            self.model_name = env_model
        
        # Processing settings
        if os.getenv("LITERATURE_MAPPER_MAX_FILE_SIZE"):
            try:
                self.max_file_size = int(os.getenv("LITERATURE_MAPPER_MAX_FILE_SIZE"))
            except ValueError:
                logger.warning("Invalid LITERATURE_MAPPER_MAX_FILE_SIZE, using default")
        
        if os.getenv("LITERATURE_MAPPER_BATCH_SIZE"):
            try:
                self.batch_size = int(os.getenv("LITERATURE_MAPPER_BATCH_SIZE"))
            except ValueError:
                logger.warning("Invalid LITERATURE_MAPPER_BATCH_SIZE, using default")
        
        # Logging
        env_log_level = os.getenv("LITERATURE_MAPPER_LOG_LEVEL")
        if env_log_level:
            self.log_level = env_log_level.upper()
        
        if os.getenv("LITERATURE_MAPPER_VERBOSE", "").lower() in ("true", "1", "yes"):
            self.verbose = True
    
    def _validate_config(self):
        """Validate configuration values."""
        # Basic model name validation
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValidationError(
                f"Invalid model_name: {self.model_name}",
                field="model_name"
            )
        
        # File size validation
        if self.max_file_size <= 0 or self.max_file_size > 500 * 1024 * 1024:
            raise ValidationError(
                f"Invalid max_file_size: {self.max_file_size}",
                field="max_file_size"
            )
        
        # Batch size validation
        if self.batch_size <= 0 or self.batch_size > 1000:
            raise ValidationError(
                f"Invalid batch_size: {self.batch_size}",
                field="batch_size"
            )
        
        # Retry validation
        if self.max_retries < 0 or self.max_retries > 10:
            raise ValidationError(
                f"Invalid max_retries: {self.max_retries}",
                field="max_retries"
            )
        
        # Log level validation
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValidationError(
                f"Invalid log_level: {self.log_level}",
                field="log_level"
            )
    
    def get_model_type(self) -> str:
        """
        Get model type from model name for optimization.
        
        Returns:
            Model type: "flash", "pro", "ultra", or "unknown"
        """
        name_lower = self.model_name.lower()
        
        if "flash" in name_lower:
            return "flash"
        elif "pro" in name_lower:
            return "pro"
        elif "ultra" in name_lower:
            return "ultra"
        else:
            return "unknown"
    
    def to_dict(self) -> dict[str, any]:
        """Convert configuration to dictionary for logging."""
        return {
            "api_key": "***" if self.api_key else None,
            "model_name": self.model_name,
            "model_type": self.get_model_type(),
            "max_file_size": self.max_file_size,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "log_level": self.log_level,
            "verbose": self.verbose
        }

def load_config(**overrides) -> LiteratureMapperConfig:
    """
    Load configuration with optional overrides.
    
    Args:
        **overrides: Configuration values to override
        
    Returns:
        LiteratureMapperConfig instance
    """
    return LiteratureMapperConfig(**overrides)

def validate_environment() -> list[str]:
    """
    Validate environment and return any issues.
    
    Returns:
        List of warning/error messages
    """
    issues = []
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, got {sys.version}")
    
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        issues.append("GEMINI_API_KEY environment variable not set")
    
    # Check optional dependencies
    optional_deps = {
        "python-magic": "Enhanced file type detection",
        "PyYAML": "YAML configuration support"
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep.replace("-", "_").lower())
        except ImportError:
            issues.append(f"Optional dependency '{dep}' not found ({description})")
    
    return issues

# Export main components
__all__ = [
    'LiteratureMapperConfig',
    'load_config',
    'validate_environment',
    'DEFAULT_MODEL',
    'FALLBACK_MODEL',
    'VERSION'
]