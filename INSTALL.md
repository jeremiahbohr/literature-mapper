# Installation Guide

## Quick Installation

```bash
# Install from PyPI (when published)
pip install literature-mapper

# Or install from GitHub
pip install git+https://github.com/johannes-bohr/literature-mapper.git
```

## Enhanced Installation (Recommended)

For better file type detection:

```bash
pip install "literature-mapper[enhanced]"
```

This adds `python-magic` for more reliable PDF validation.

## Development Installation

For contributors:

```bash
# Clone and install in development mode
git clone https://github.com/johannes-bohr/literature-mapper.git
cd literature-mapper
pip install -e ".[dev,enhanced]"
```

## API Key Setup

**Required:** Get a Google AI API key and set it as an environment variable.

1. Get your API key at [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Make it permanent:
```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
echo 'export GEMINI_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

## Verify Installation

```bash
# Check CLI works
literature-mapper --help

# Test Python import
python -c "from literature_mapper import LiteratureMapper; print('✓ Installation successful!')"

# Check models (requires API key)
literature-mapper models
```

## Optional Configuration

Set additional environment variables for customization:

```bash
# Use a different model
export LITERATURE_MAPPER_MODEL="gemini-2.5-pro"

# Increase file size limit (in bytes)
export LITERATURE_MAPPER_MAX_FILE_SIZE="100000000"  # 100MB

# Enable verbose logging
export LITERATURE_MAPPER_VERBOSE="true"
export LITERATURE_MAPPER_LOG_LEVEL="DEBUG"
```

## System Requirements

- **Python 3.8 or higher**
- **Internet connection** for AI API calls
- **~10MB disk space** for the package
- **Write access** to directories where you want to create literature databases

## Dependencies

### Core (automatically installed):
- `google-generativeai` - AI model access
- `pandas` - Data manipulation
- `sqlalchemy` - Database operations
- `pypdf` - PDF text extraction
- `typer` + `rich` - CLI interface
- `tqdm` - Progress bars

### Optional (install with `[enhanced]`):
- `python-magic` - Better file type detection (Linux/macOS)
- `PyYAML` - YAML support (unused but may be useful for custom scripts)

## Troubleshooting

### Common Issues

**Import Error:**
```bash
# Check Python version
python --version  # Should be 3.8+
```

**API Key Error:**
```bash
# Verify key is set
echo $GEMINI_API_KEY

# Test API access
literature-mapper models
```

**Permission Errors:**
```bash
# Use virtual environment (recommended)
python -m venv literature-env
source literature-env/bin/activate  # Linux/Mac
# literature-env\Scripts\activate    # Windows
pip install literature-mapper
```

**PDF Processing Issues:**
- Ensure PDFs are not password-protected
- Check file size limits (default 50MB)
- Verify files are actual PDFs, not scanned images

### Getting Help

- Check logs in your corpus directory: `literature_mapper.log`
- Use verbose mode: `literature-mapper process ./corpus --verbose`
- Enable debug logging: `export LITERATURE_MAPPER_LOG_LEVEL="DEBUG"`

### Platform Notes

**Linux/macOS:** Enhanced file detection works best  
**Windows:** Basic functionality works fine; enhanced features may require WSL  
**Apple Silicon:** All features supported