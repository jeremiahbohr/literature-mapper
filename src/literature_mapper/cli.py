import typer
from typing import Optional
from pathlib import Path
import os
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import pandas as pd

# Import our modules
try:
    from .mapper import LiteratureMapper
    from .database import get_database_info
    from .exceptions import ValidationError, DatabaseError, APIError, PDFProcessingError
    from .validation import validate_api_key, validate_directory_path
    from .config import DEFAULT_MODEL
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this as part of the literature-mapper package")
    raise typer.Exit(1)

console = Console()
app = typer.Typer(
    name="literature-mapper",
    help="AI-powered literature analysis and database creation tool",
    rich_markup_mode="rich"
)

def handle_error(e: Exception) -> None:
    """Convert exceptions to user-friendly messages and exit."""
    if hasattr(e, 'user_message'):
        console.print(f"[red]{e.user_message}[/red]")
    else:
        console.print(f"[red]Error: {e}[/red]")
    raise typer.Exit(1)

def validate_inputs(corpus_path: str) -> Path:
    """Validate and return corpus directory path."""
    try:
        path = Path(corpus_path).resolve()
        if not validate_directory_path(path, check_writable=True):
            console.print(f"[red]Invalid directory path: {path}[/red]")
            raise typer.Exit(1)
        return path
    except Exception as e:
        handle_error(e)

def setup_api_key() -> str:
    """Validate API key setup."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]GEMINI_API_KEY environment variable not set.[/red]")
        console.print("Set your API key: export GEMINI_API_KEY='your_api_key_here'")
        raise typer.Exit(1)
    
    if not validate_api_key(api_key):
        console.print("[red]Invalid API key format.[/red]")
        raise typer.Exit(1)
        
    return api_key

@app.command()
def process(
    corpus_path: str = typer.Argument(..., help="Path to the research corpus directory"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Gemini model to use"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Include PDFs in subfolders"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Process new PDF papers in the corpus directory."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    corpus_path_obj = validate_inputs(corpus_path)
    api_key = setup_api_key()
    
    # Check for PDF files
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(corpus_path_obj.glob(pattern))
    if not pdf_files:
        folder_desc = "directory tree" if recursive else "directory"
        console.print(f"[yellow]No PDF files found in {folder_desc} {corpus_path}[/yellow]")
        console.print("Add some PDF files and try again.")
        return
    
    folder_desc = "directory tree" if recursive else "directory"
    console.print(f"[green]Found {len(pdf_files)} PDF files in {folder_desc}[/green]")
    
    # Process papers
    try:
        mapper = LiteratureMapper(str(corpus_path_obj), model_name=model, api_key=api_key)
        
        with Progress(SpinnerColumn(), TextColumn("Processing papers..."), console=console) as progress:
            task = progress.add_task("Processing...", total=None)
            results = mapper.process_new_papers(recursive=recursive)
            progress.update(task, completed=True)
        
        # Display results
        console.print("\n[bold green]Processing Complete![/bold green]")
        
        table = Table(title="Results")
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        
        table.add_row("Processed", str(results.processed))
        table.add_row("Failed", str(results.failed))
        table.add_row("Skipped", str(results.skipped))
        
        console.print(table)
        
        if results.failed > 0:
            console.print("[yellow]Check logs for details on failed papers[/yellow]")
        
        if results.processed > 0:
            console.print(f"[green]Successfully processed {results.processed} papers![/green]")
            
    except (ValidationError, DatabaseError, APIError, PDFProcessingError) as e:
        handle_error(e)
        
@app.command()
def export(
    corpus_path: str = typer.Argument(..., help="Path to the research corpus directory"),
    output_path: str = typer.Argument(..., help="Path for the output CSV file"),
):
    """Export the corpus database to a CSV file."""
    corpus_path_obj = validate_inputs(corpus_path)
    
    # Check if database exists
    db_info = get_database_info(corpus_path_obj)
    if not db_info.exists:
        console.print(f"[red]No database found at {corpus_path}[/red]")
        console.print("Run 'literature-mapper process' first.")
        raise typer.Exit(1)
    
    try:
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        console.print(f"[red]Invalid output path: {e}[/red]")
        raise typer.Exit(1)
    
    try:
        # No need to pass api_key for export-only operations
        mapper = LiteratureMapper(str(corpus_path_obj), model_name=DEFAULT_MODEL)
        
        with Progress(SpinnerColumn(), TextColumn("Exporting..."), console=console) as progress:
            task = progress.add_task("Exporting data...", total=None)
            mapper.export_to_csv(str(output_file))
            progress.update(task, completed=True)
        
        file_size = output_file.stat().st_size / 1024  # KB
        console.print(f"[green]Export successful![/green]")
        console.print(f"File: {output_file} ({file_size:.1f} KB)")
        
    except (ValidationError, DatabaseError) as e:
        handle_error(e)

@app.command()
def status(corpus_path: str = typer.Argument(..., help="Path to the research corpus directory")):
    """Show corpus status and statistics."""
    corpus_path_obj = validate_inputs(corpus_path)
    
    console.print(f"[bold blue]Corpus Status[/bold blue]")
    console.print(f"Directory: {corpus_path_obj}")
    
    # Count PDF files
    pdf_files = list(corpus_path_obj.glob("*.pdf"))
    console.print(f"PDF Files: {len(pdf_files)}")
    
    # Database info
    db_info = get_database_info(corpus_path_obj)
    
    if db_info.exists:
        console.print(f"Database: {db_info.size_mb} MB")
        
        if db_info.table_counts:
            table = Table(title="Database Contents")
            table.add_column("Table", style="cyan")
            table.add_column("Records", justify="right", style="magenta")
            
            for table_name, count in db_info.table_counts.items():
                table.add_row(table_name.title(), str(count))
            
            console.print(table)
            
            # Show processing efficiency
            processed_count = db_info.table_counts.get('papers', 0)
            if len(pdf_files) > 0:
                efficiency = (processed_count / len(pdf_files)) * 100
                console.print(f"Processing: {efficiency:.0f}% ({processed_count}/{len(pdf_files)} PDFs)")
        else:
            console.print("[yellow]Database exists but is empty[/yellow]")
    else:
        console.print("[red]No database found[/red]")
        console.print("Run 'literature-mapper process' to create it.")

@app.command()
def models(
    details: bool = typer.Option(False, "--details", "-d", help="Show detailed model information")
):
    """List available Gemini models."""
    api_key = setup_api_key()
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        console.print("[blue]Fetching available models...[/blue]")
        models_list = genai.list_models()
        
        gemini_models = []
        for model in models_list:
            if hasattr(model, 'name'):
                model_name = model.name.split('/')[-1] if '/' in model.name else model.name
                if 'gemini' in model_name.lower():
                    gemini_models.append(model_name)
        
        if not gemini_models:
            console.print("[red]No Gemini models found[/red]")
            return
        
        gemini_models.sort()
        
        if details:
            table = Table(title="Available Gemini Models")
            table.add_column("Model Name", style="cyan")
            table.add_column("Best For", style="yellow")
            
            for model in gemini_models:
                # Simplified model recommendations without over-engineering
                if "flash" in model.lower():
                    recommendation = "Fast analysis, large batches"
                elif "pro" in model.lower():
                    recommendation = "Balanced analysis, most use cases"
                else:
                    recommendation = "General purpose"
                
                marker = " (default)" if model == DEFAULT_MODEL else ""
                table.add_row(model + marker, recommendation)
            
            console.print(table)
        else:
            console.print(f"[green]Found {len(gemini_models)} models:[/green]")
            for model in gemini_models:
                marker = " (default)" if model == DEFAULT_MODEL else ""
                console.print(f"  • {model}{marker}")
        
        console.print(f"\nUse --model option to specify: --model {gemini_models[1] if len(gemini_models) > 1 else gemini_models[0]}")
        
    except Exception as e:
        console.print(f"[red]Failed to fetch models: {e}[/red]")
        console.print(f"Try using default model: {DEFAULT_MODEL}")

@app.command()
def papers(
    corpus_path: str = typer.Argument(..., help="Path to the research corpus directory"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of papers to show"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Filter by year"),
):
    """List papers in the corpus."""
    corpus_path_obj = validate_inputs(corpus_path)
    
    # Check if database exists
    db_info = get_database_info(corpus_path_obj)
    if not db_info.exists:
        console.print(f"[red]No database found at {corpus_path}[/red]")
        raise typer.Exit(1)
    
    try:
        # No need for API key for read-only operations
        mapper = LiteratureMapper(str(corpus_path_obj), model_name=DEFAULT_MODEL)
        df = mapper.get_all_analyses()
        
        if df.empty:
            console.print("[yellow]No papers found in database[/yellow]")
            return
        
        # Apply filters
        if year:
            df = df[df['year'] == year]
            if df.empty:
                console.print(f"[yellow]No papers found for year {year}[/yellow]")
                return
        
        # Sort and limit
        df = df.sort_values('year', ascending=False).head(limit)
        
        # Display table
        table = Table(title=f"Papers ({len(df)} shown)")
        table.add_column("Year", style="blue", width=6)
        table.add_column("Title", style="green", width=45)
        table.add_column("Authors", style="yellow", width=25)
        
        for _, row in df.iterrows():
            title = (row['title'][:42] + "...") if len(str(row['title'])) > 45 else str(row['title'])
            authors = (row['authors'][:22] + "...") if len(str(row['authors'])) > 25 else str(row['authors'])
            
            table.add_row(
                str(row['year']) if pd.notna(row['year']) else "N/A",
                title,
                authors if pd.notna(row['authors']) else "N/A"
            )
        
        console.print(table)
        
        total_papers = len(mapper.get_all_analyses())
        if len(df) < total_papers:
            console.print(f"Showing {len(df)} of {total_papers} total papers")
            
    except (ValidationError, DatabaseError) as e:
        handle_error(e)

if __name__ == "__main__":
    app()