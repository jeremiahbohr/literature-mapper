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
    from .ai_prompts import get_analysis_prompt, get_kg_prompt
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
        if not validate_directory_path(path):
            console.print(f"[red]Invalid directory path: {path}[/red]")
            raise typer.Exit(1)
        return path
    except Exception as e:
        handle_error(e)

def setup_api_key(required: bool = True) -> Optional[str]:
    """Validate API key setup."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        if required:
            console.print("[red]GEMINI_API_KEY environment variable not set.[/red]")
            console.print("Set your API key: export GEMINI_API_KEY='your_api_key_here'")
            raise typer.Exit(1)
        return None
    
    if required and not validate_api_key(api_key):
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
        mapper = LiteratureMapper(str(corpus_path_obj), model_name=DEFAULT_MODEL, validate_api=False)
        
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
                console.print(f"  - {model}{marker}")
        
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
        mapper = LiteratureMapper(str(corpus_path_obj), model_name=DEFAULT_MODEL, validate_api=False)
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

@app.command()
def viz(
    corpus_path: str = typer.Argument(..., help="Path to the research corpus directory"),
    output: str = typer.Option("graph.gexf", "--output", "-o", help="Output GEXF file path"),
    threshold: float = typer.Option(0.1, "--threshold", "-t", help="Proportional threshold (0.0-1.0) for edge consensus"),
    mode: str = typer.Option("semantic", "--mode", "-m", help="Graph mode: semantic, authors, concepts, river, similarity"),
):
    """
    Export the Knowledge Graph to GEXF format for Gephi.
    
    Modes:
    - semantic: The core Knowledge Graph (Concepts, Findings, etc.)
    - authors: Co-authorship network (Invisible College)
    - concepts: Topic co-occurrence network
    - river: Dynamic concept network over time
    - similarity: Paper similarity network based on shared concepts
    
    The threshold determines the minimum consensus required for an edge to appear.
    """
    corpus_path_obj = validate_inputs(corpus_path)
    
    # Check if database exists
    db_info = get_database_info(corpus_path_obj)
    if not db_info.exists:
        console.print(f"[red]No database found at {corpus_path}[/red]")
        console.print("Run 'literature-mapper process' first.")
        raise typer.Exit(1)
        
    try:
        from .viz import export_to_gexf
        
        output_path = Path(output).resolve()
        
        with Progress(SpinnerColumn(), TextColumn(f"Exporting {mode} graph..."), console=console) as progress:
            task = progress.add_task("Exporting...", total=None)
            export_to_gexf(str(corpus_path_obj), str(output_path), threshold=threshold, mode=mode)
            progress.update(task, completed=True)
            
        console.print(f"[green]Successfully exported {mode} graph to {output_path}[/green]")
        console.print(f"Threshold: {threshold}")
        
    except ImportError:
        console.print("[red]Visualization module not found. Reinstall package.[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        handle_error(e)

@app.command()
def cost(
    corpus_path: str = typer.Argument(..., help="Path to the research corpus directory"),
    model: str = typer.Option(..., "--model", "-m", help="Gemini model to use (Required)"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Include PDFs in subfolders"),
):
    """Estimate the cost of processing the corpus."""
    corpus_path_obj = validate_inputs(corpus_path)
    api_key = setup_api_key()
    
    # Check for PDF files
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(corpus_path_obj.glob(pattern))
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {corpus_path}[/yellow]")
        return
    
    console.print(f"[bold blue]Estimating cost for {len(pdf_files)} papers using {model}...[/bold blue]")
    
    try:
        mapper = LiteratureMapper(str(corpus_path_obj), model_name=model, api_key=api_key)
        
        total_input_tokens = 0
        processed_count = 0
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Counting tokens...", total=len(pdf_files))
            
            for pdf_path in pdf_files:
                try:
                    # Extract text
                    text = mapper.pdf_processor.extract_text(pdf_path)
                    
                    # Construct prompts (Analysis + KG)
                    analysis_prompt = get_analysis_prompt().format(text=text[:50000])
                    kg_prompt = get_kg_prompt(text=text[:50000])
                    
                    # Count tokens
                    input_tokens = mapper.ai_analyzer.count_tokens(analysis_prompt) + mapper.ai_analyzer.count_tokens(kg_prompt)
                    total_input_tokens += input_tokens
                    processed_count += 1
                    
                except Exception as e:
                    # Skip failed files in estimate
                    pass
                
                progress.advance(task)
        
        # Estimates
        # Input cost: $0.075 / 1M tokens (Flash)
        # Output cost: $0.30 / 1M tokens (Flash) - Estimate 2k output tokens per paper
        
        avg_output_tokens = 2000
        total_output_tokens = processed_count * avg_output_tokens
        
        # Pricing Table (USD per 1M tokens)
        # Source: Google Cloud Pricing (approximate)
        PRICING = {
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-1.5-pro":   {"input": 1.25,  "output": 5.00},
            "gemini-2.5-pro":   {"input": 1.25,  "output": 5.00},
            "gemini-1.0-pro":   {"input": 0.50,  "output": 1.50},
            "gemini-3":         {"input": 2.00,  "output": 12.00},
            "gemini-3-pro":     {"input": 2.00,  "output": 12.00},
        }
        
        # Find best match
        model_key = "unknown"
        for key in PRICING:
            if key in model.lower():
                model_key = key
                break
        
        if model_key != "unknown":
            input_price_per_m = PRICING[model_key]["input"]
            output_price_per_m = PRICING[model_key]["output"]
        else:
            # Fallback heuristics
            if "flash" in model.lower():
                input_price_per_m = 0.075
                output_price_per_m = 0.30
            else:
                # Assume Pro/Standard pricing for unknown models to be safe
                input_price_per_m = 1.25
                output_price_per_m = 5.00
                console.print(f"[yellow]Warning: Exact pricing for '{model}' not found. Using standard Pro rates.[/yellow]")
        
        input_cost = (total_input_tokens / 1_000_000) * input_price_per_m
        output_cost = (total_output_tokens / 1_000_000) * output_price_per_m
        total_cost = input_cost + output_cost
        
        console.print("\n[bold green]Estimation Complete![/bold green]")
        
        table = Table(title=f"Cost Estimate ({model})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")
        
        table.add_row("Papers", str(processed_count))
        table.add_row("Total Input Tokens", f"{total_input_tokens:,}")
        table.add_row("Est. Output Tokens", f"{total_output_tokens:,}")
        table.add_row("Est. Total Cost", f"${total_cost:.4f}")
        
        console.print(table)
        console.print("[dim]Note: Output tokens are estimated at 2,000 per paper. Actual costs may vary.[/dim]")
            
    except Exception as e:
        handle_error(e)

@app.command()
def ghosts(
    corpus_path: str = typer.Argument(..., help="Path to the research corpus directory"),
    mode: str = typer.Option("bibliographic", "--mode", "-m", help="Mode: 'bibliographic', 'authors'"),
    threshold: int = typer.Option(3, "--threshold", "-t", help="Minimum citations to be considered a ghost"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results to display"),
):
    """Hunt for Ghost Nodes (missing papers or authors in the corpus)."""
    corpus_path_obj = validate_inputs(corpus_path)
    
    try:
        from .ghosts import GhostHunter
        
        mapper = LiteratureMapper(str(corpus_path_obj), model_name=DEFAULT_MODEL, validate_api=False)
        hunter = GhostHunter(mapper)

        if mode == "bibliographic":
            console.print(f"[bold blue]Hunting for Bibliographic Ghosts (Threshold: {threshold})...[/bold blue]")
            df = hunter.find_bibliographic_ghosts(threshold=threshold)
            
            if df.empty:
                console.print("[yellow]No ghost nodes found matching criteria.[/yellow]")
                return
            
            df = df.head(limit)
            
            table = Table(title=f"Bibliographic Ghost Nodes (Missing Papers)")
            table.add_column("Count", style="magenta", justify="right")
            table.add_column("Year", style="cyan")
            table.add_column("Author", style="green")
            table.add_column("Title", style="white")
            
            for _, row in df.iterrows():
                table.add_row(
                    str(row['citation_count']),
                    str(row['year']) if row['year'] else "?",
                    str(row['author']) if row['author'] else "?",
                    row['title']
                )
            
            console.print(table)
            console.print(f"\n[dim]Found {len(df)} papers cited frequently but missing from corpus. Use --limit to change output length[/dim]")
            
        elif mode == "authors":
            console.print(f"[bold blue]Hunting for Missing Authors (Threshold: {threshold})...[/bold blue]")
            df = hunter.find_missing_authors(threshold=threshold)
            
            if df.empty:
                console.print("[yellow]No missing authors found matching criteria.[/yellow]")
                return
            
            df = df.head(limit)
            
            table = Table(title=f"Missing Authors (Frequently Cited but Not in Corpus)")
            table.add_column("Cited By", style="magenta", justify="right")
            table.add_column("Author", style="cyan")
            table.add_column("Sample Works", style="green")
            
            for _, row in df.iterrows():
                table.add_row(
                    str(row['cited_by_papers']),
                    row['author'],
                    row['sample_works'][:80] + "..." if len(row['sample_works']) > 80 else row['sample_works']
                )
            
            console.print(table)
            console.print(f"\n[dim]Found {len(df)} authors frequently cited but not represented in corpus. Use --limit to change output length.[/dim]")

        else:
            console.print(f"[bold red]Error:[/bold red] Mode '{mode}' not recognized.")
            console.print("Available modes: [green]bibliographic[/green], [green]authors[/green]")
            
    except Exception as e:
        handle_error(e)

@app.command()
def reset(
    corpus_path: str = typer.Argument(..., help="Path to the research corpus directory"),
    target: str = typer.Option("citations", "--target", "-t", help="Target to reset: 'citations' or 'all'"),
):
    """Reset specific parts of the database (e.g., clear citations to re-extract)."""
    corpus_path_obj = validate_inputs(corpus_path)
    
    if target not in ["citations", "all"]:
        console.print(f"[red]Invalid target: {target}. Must be 'citations' or 'all'[/red]")
        raise typer.Exit(1)
        
    confirm = typer.confirm(f"Are you sure you want to delete {target} from the database?")
    if not confirm:
        console.print("Aborted.")
        return

    try:
        from .database import get_db_session, Citation, Base
        import sqlalchemy as sa
        
        with get_db_session(corpus_path_obj) as session:
            if target == "citations":
                # Delete all citations
                count = session.query(Citation).delete()
                console.print(f"[green]Deleted {count} citations.[/green]")
                
                # Also reset citation_count on papers
                session.execute(sa.text("UPDATE papers SET citation_count = NULL"))
                console.print("[green]Reset citation counts on papers.[/green]")
                
            elif target == "all":
                # Drop all tables and recreate
                Base.metadata.drop_all(session.get_bind())
                Base.metadata.create_all(session.get_bind())
                console.print("[green]Database completely reset.[/green]")
                
            session.commit()
            
    except Exception as e:
        handle_error(e)

@app.command()
def synthesize(
    query: str = typer.Argument(..., help="Research question or topic to synthesize"),
    corpus_path: str = typer.Option(".", "--corpus", "-c", help="Path to the research corpus directory"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Gemini model to use"),
):
    """Synthesize research findings using the Argument Agent (RAG)."""
    corpus_path_obj = validate_inputs(corpus_path)
    api_key = setup_api_key()
    
    try:
        mapper = LiteratureMapper(str(corpus_path_obj), model_name=model, api_key=api_key)
        
        console.print(f"[bold blue]Synthesizing answer for: '{query}'[/bold blue]")
        with Progress(SpinnerColumn(), TextColumn("Analyzing corpus..."), console=console) as progress:
            task = progress.add_task("Thinking...", total=None)
            response = mapper.synthesize_answer(query)
            progress.update(task, completed=True)
            
        console.print("\n[bold green]Synthesis:[/bold green]")
        console.print(response)
        
    except Exception as e:
        handle_error(e)

@app.command()
def validate(
    hypothesis: str = typer.Argument(..., help="Hypothesis or claim to validate"),
    corpus_path: str = typer.Option(".", "--corpus", "-c", help="Path to the research corpus directory"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Gemini model to use"),
):
    """Validate a hypothesis using the Validation Agent."""
    corpus_path_obj = validate_inputs(corpus_path)
    api_key = setup_api_key()
    
    try:
        mapper = LiteratureMapper(str(corpus_path_obj), model_name=model, api_key=api_key)
        
        console.print(f"[bold blue]Validating hypothesis: '{hypothesis}'[/bold blue]")
        with Progress(SpinnerColumn(), TextColumn("Critiquing against evidence..."), console=console) as progress:
            task = progress.add_task("Thinking...", total=None)
            result = mapper.validate_hypothesis(hypothesis)
            progress.update(task, completed=True)
            
        console.print(f"\n[bold]Verdict:[/bold] {result.get('verdict', 'UNKNOWN')}")
        console.print(f"[bold]Explanation:[/bold] {result.get('explanation', 'No explanation provided.')}")
        
        if result.get('citations'):
            console.print("\n[bold]Evidence:[/bold]")
            for cit in result['citations']:
                console.print(f"- {cit}")
        
    except Exception as e:
        handle_error(e)



@app.command()
def citations(
    corpus_path: str = typer.Argument(..., help="Path to the research corpus directory"),
    email: str = typer.Option(None, "--email", "-e", help="Email for OpenAlex polite pool (faster rate limits)"),
):
    """
    Fetch citations and reference data from OpenAlex.
    
    This looks up each paper in your corpus on OpenAlex and retrieves:
    - Citation count (how many times the paper has been cited)
    - References (papers cited by your paper)
    """
    corpus_path_obj = validate_inputs(corpus_path)
    
    # Check if database exists
    db_info = get_database_info(corpus_path_obj)
    if not db_info.exists:
        console.print(f"[red]No database found at {corpus_path}[/red]")
        console.print("Run 'literature-mapper process' first.")
        raise typer.Exit(1)
    
    try:
        from .openalex import fetch_citations_for_corpus
        
        stats = fetch_citations_for_corpus(str(corpus_path_obj), email=email)
        
        if stats['not_found'] > 0:
            console.print(f"\n[yellow]Note: {stats['not_found']} papers were not found in OpenAlex.[/yellow]")
            console.print("[dim]This can happen with very recent papers, preprints, or obscure venues.[/dim]")
        
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Run: pip install requests")
        raise typer.Exit(1)
    except Exception as e:
        handle_error(e)

@app.command()
def hubs(
    corpus_path: str = typer.Argument(..., help="Path to the research corpus directory"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of papers to show"),
):
    """Show the most influential papers in the corpus by citation count."""
    corpus_path_obj = validate_inputs(corpus_path)
    
    # Check if database exists
    db_info = get_database_info(corpus_path_obj)
    if not db_info.exists:
        console.print(f"[red]No database found at {corpus_path}[/red]")
        console.print("Run 'literature-mapper process' first.")
        raise typer.Exit(1)
    
    try:
        from .analysis import CorpusAnalyzer
        
        analyzer = CorpusAnalyzer(corpus_path_obj)
        df = analyzer.find_hub_papers(limit=limit)
        
        if df.empty:
            console.print("[yellow]No citation counts available.[/yellow]")
            console.print("Run 'literature-mapper citations' first to fetch citation data.")
            return
        
        table = Table(title=f"Hub Papers (Top {len(df)} by Citation Count)")
        table.add_column("Citations", style="magenta", justify="right")
        table.add_column("Year", style="cyan")
        table.add_column("Authors", style="green")
        table.add_column("Title", style="white")
        
        for _, row in df.iterrows():
            title = row['title'][:60] + "..." if len(row['title']) > 60 else row['title']
            table.add_row(
                f"{row['citation_count']:,}",
                str(row['year']),
                row['authors'][:30] + "..." if len(row['authors']) > 30 else row['authors'],
                title
            )
        
        console.print(table)
        console.print(f"\n[dim]These are the most-cited papers in your corpus. Use --limit to change output length.[/dim]")
        
    except Exception as e:
        handle_error(e)

@app.command()
def stats(
    corpus_path: str = typer.Argument(..., help="Path to the research corpus directory"),
):
    """Show comprehensive corpus statistics."""
    corpus_path_obj = validate_inputs(corpus_path)
    
    # Check if database exists
    db_info = get_database_info(corpus_path_obj)
    if not db_info.exists:
        console.print(f"[red]No database found at {corpus_path}[/red]")
        console.print("Run 'literature-mapper process' first.")
        raise typer.Exit(1)
    
    try:
        from .analysis import CorpusAnalyzer
        
        analyzer = CorpusAnalyzer(corpus_path_obj)
        stats = analyzer.get_statistics()
        
        console.print(f"\n[bold blue]Corpus Statistics[/bold blue]")
        console.print(f"Directory: {corpus_path_obj}")
        
        table = Table(title="Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")
        
        table.add_row("Total Papers", str(stats.total_papers))
        table.add_row("Total Authors", str(stats.total_authors))
        table.add_row("Total Concepts", str(stats.total_concepts))
        table.add_row("Papers with Citations", str(stats.papers_with_citations))
        table.add_row("Total Citations", f"{stats.total_citations:,}")
        
        if stats.year_range:
            table.add_row("Year Range", f"{stats.year_range[0]} - {stats.year_range[1]}")
        
        console.print(table)
        
        if stats.top_journals:
            journal_table = Table(title="Top Journals/Venues")
            journal_table.add_column("Journal", style="green")
            journal_table.add_column("Papers", justify="right", style="magenta")
            
            for journal, count in stats.top_journals[:10]:
                journal_table.add_row(journal, str(count))
            
            console.print(journal_table)
        
    except Exception as e:
        handle_error(e)

if __name__ == "__main__":
    app()