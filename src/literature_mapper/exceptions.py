"""
Custom exception classes for Literature Mapper.
"""

from typing import Optional, Dict, Any
from pathlib import Path

class LiteratureMapperError(Exception):
    """
    Base exception for all Literature Mapper errors.
    """
    
    def __init__(
        self, 
        message: str, 
        user_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.user_message = user_message or self._generate_user_message()
    
    def _generate_user_message(self) -> str:
        """Generate a user-friendly error message."""
        return "An error occurred during literature processing. Check the logs for details."
    
    def __str__(self) -> str:
        """String representation for logging."""
        if self.context:
            context_str = ', '.join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class ValidationError(LiteratureMapperError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        context = kwargs.get('context', {})
        if field:
            context['field'] = field
        if value is not None:
            context['value'] = str(value)[:50]  # Truncate long values
        kwargs['context'] = context
        super().__init__(message, **kwargs)
    
    def _generate_user_message(self) -> str:
        field = self.context.get('field')
        if field:
            return f"Invalid {field}: {self.message}"
        return f"Validation error: {self.message}"


class PDFProcessingError(LiteratureMapperError):
    """Raised when PDF processing fails."""
    
    def __init__(
        self, 
        message: str, 
        pdf_path: Optional[Path] = None,
        error_type: str = "unknown",
        **kwargs
    ):
        context = kwargs.get('context', {})
        if pdf_path:
            context['pdf_name'] = pdf_path.name
        context['error_type'] = error_type
        kwargs['context'] = context
        super().__init__(message, **kwargs)
    
    def _generate_user_message(self) -> str:
        error_type = self.context.get('error_type', 'unknown')
        pdf_name = self.context.get('pdf_name', 'file')
        
        messages = {
            'validation': f"File '{pdf_name}' is not a valid PDF or is too large.",
            'encryption': f"File '{pdf_name}' is password-protected. Remove the password and try again.",
            'extraction': f"Could not extract text from '{pdf_name}'. This may be a scanned document.",
            'corruption': f"File '{pdf_name}' appears to be corrupted.",
        }
        
        return messages.get(error_type, f"Failed to process '{pdf_name}': {self.message}")


class APIError(LiteratureMapperError):
    """Raised when external API calls fail."""
    
    def __init__(
        self, 
        message: str, 
        api_name: str = "Gemini API",
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['api_name'] = api_name
        kwargs['context'] = context
        super().__init__(message, **kwargs)
    
    def _generate_user_message(self) -> str:
        api_name = self.context.get('api_name', 'API')
        
        # Categorize errors by message content
        if any(keyword in self.message.lower() for keyword in ['authentication', 'api key', 'unauthorized']):
            return f"{api_name} authentication failed. Please check your API key."
        elif any(keyword in self.message.lower() for keyword in ['rate limit', 'quota', 'too many requests']):
            return f"{api_name} rate limit exceeded. Please try again later."
        elif 'timeout' in self.message.lower():
            return f"{api_name} request timed out. Please try again."
        elif 'json' in self.message.lower():
            return f"{api_name} returned invalid data. Please try again."
        else:
            return f"{api_name} error: {self.message}"


class DatabaseError(LiteratureMapperError):
    """Raised when database operations fail."""
    
    def _generate_user_message(self) -> str:
        if 'UNIQUE constraint failed' in self.message:
            return "A paper with this title and year already exists."
        elif 'database is locked' in self.message:
            return "Database is in use. Please try again in a moment."
        elif 'permission denied' in self.message:
            return "Cannot access database. Please check file permissions."
        elif 'no such table' in self.message:
            return "Database is corrupted or outdated. Please recreate it."
        else:
            return f"Database error: {self.message}"


# Export all exception classes
__all__ = [
    'LiteratureMapperError',
    'ValidationError',
    'PDFProcessingError',
    'APIError',
    'DatabaseError'
]