"""
Command-line interface for speedX.
"""
import click
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from .mast_client import MASTClient
from .classifier import RAGBoostClassifier
from .features import get_default_featurizer
from .fits_utils import extract_fits_sketch, sketch_to_features


@click.group()
@click.version_option(version="0.1.0")
def main():
    """
    speedX: High-throughput science-content classifier for HST data.
    
    A tool for querying HST observations via MAST API and performing
    fast inference with retrieval-augmented classification.
    """
    pass


@main.command()
@click.option(
    "--instrument",
    "-i",
    help="Filter by instrument name (e.g., 'ACS/WFC', 'WFC3/UVIS')"
)
@click.option(
    "--target",
    "-t",
    help="Filter by target name"
)
@click.option(
    "--proposal",
    "-p",
    help="Filter by proposal ID"
)
@click.option(
    "--max-records",
    "-n",
    type=int,
    default=1000,
    help="Maximum number of records to retrieve"
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output file path (parquet or csv)"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["parquet", "csv"]),
    default="parquet",
    help="Output format"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output"
)
def query_metadata(instrument, target, proposal, max_records, output, format, verbose):
    """
    Query HST observation metadata from MAST and save to file.
    
    Example:
        speedx query-metadata --instrument "ACS/WFC" --max-records 100 -o data.parquet
    """
    try:
        if verbose:
            click.echo("Initializing MAST client...")
        
        click.echo("Querying MAST for HST observations...")
        
        client = MASTClient()
        
        observations = client.query_observations(
            instrument=instrument,
            target_name=target,
            proposal_id=proposal,
            max_records=max_records
        )
        
        click.echo(f"Found {len(observations)} observations")
        
        if not observations.empty:
            if verbose:
                click.echo(f"Saving to {output}...")
            client.save_metadata(observations, output, format=format)
            click.echo(f"✓ Metadata saved to {output}")
        else:
            click.echo("⚠ No observations found", err=True)
            
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        raise click.Abort()


@main.command()
@click.option(
    "--instrument",
    "-i",
    help="Filter by instrument name"
)
@click.option(
    "--n-samples",
    "-n",
    type=int,
    default=10,
    help="Number of samples to fetch"
)
@click.option(
    "--output-dir",
    "-o",
    default="./data/samples",
    help="Output directory for downloaded files"
)
def fetch_sample(instrument, n_samples, output_dir):
    """
    Fetch a small sample of HST observations and download products.
    
    Example:
        speedx fetch-sample --instrument "WFC3/UVIS" -n 5 -o ./samples
    """
    click.echo(f"Fetching {n_samples} sample observations...")
    
    client = MASTClient()
    
    files = client.fetch_sample(
        n_samples=n_samples,
        instrument=instrument,
        output_dir=output_dir
    )
    
    click.echo(f"Downloaded {len(files)} files to {output_dir}")
    for f in files:
        click.echo(f"  - {f}")


@main.command()
@click.option(
    "--metadata",
    "-m",
    required=True,
    help="Path to metadata file (parquet or csv)"
)
@click.option(
    "--labels",
    "-l",
    required=True,
    help="Path to labels file (csv with obs_id and label columns)"
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output path for trained model (pickle)"
)
@click.option(
    "--n-estimators",
    type=int,
    default=10,
    help="Number of boosting rounds"
)
@click.option(
    "--k-neighbors",
    type=int,
    default=5,
    help="Number of neighbors for retrieval"
)
def train(metadata, labels, output, n_estimators, k_neighbors):
    """
    Train a RAGBoostClassifier on observation metadata.
    
    Example:
        speedx train -m data.parquet -l labels.csv -o model.pkl
    """
    click.echo("Loading data...")
    
    # Load metadata
    client = MASTClient()
    df = client.load_metadata(metadata)
    
    # Load labels
    labels_df = pd.read_csv(labels)
    
    # Merge
    df = df.merge(labels_df, on="obs_id", how="inner")
    
    if df.empty:
        click.echo("Error: No matching observations found in labels file")
        return
    
    click.echo(f"Training on {len(df)} observations...")
    
    # Extract features
    featurizer = get_default_featurizer()
    X = featurizer.fit_transform(df)
    y = df["label"].values
    
    # Train classifier
    clf = RAGBoostClassifier(
        n_estimators=n_estimators,
        k_neighbors=k_neighbors,
        random_state=42
    )
    clf.fit(X, y)
    
    # Save model
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump({"classifier": clf, "featurizer": featurizer}, f)
    
    click.echo(f"Model saved to {output}")


@main.command()
@click.option(
    "--metadata",
    "-m",
    required=True,
    help="Path to metadata file (parquet or csv)"
)
@click.option(
    "--model",
    required=True,
    help="Path to trained model (pickle)"
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output path for predictions (csv)"
)
def predict(metadata, model, output):
    """
    Predict labels for observations using a trained model.
    
    Example:
        speedx predict -m data.parquet --model model.pkl -o predictions.csv
    """
    click.echo("Loading model...")
    
    with open(model, "rb") as f:
        saved = pickle.load(f)
        clf = saved["classifier"]
        featurizer = saved["featurizer"]
    
    click.echo("Loading data...")
    
    # Load metadata
    client = MASTClient()
    df = client.load_metadata(metadata)
    
    click.echo(f"Predicting labels for {len(df)} observations...")
    
    # Extract features
    X = featurizer.transform(df)
    
    # Predict
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)
    
    # Save predictions
    df_out = df[["obs_id"]].copy()
    df_out["predicted_label"] = y_pred
    
    for i, cls in enumerate(clf.classes_):
        df_out[f"proba_class_{cls}"] = y_proba[:, i]
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    
    click.echo(f"Predictions saved to {output}")


@main.command()
@click.argument("fits_path")
@click.option(
    "--output",
    "-o",
    help="Output path for sketch (json). If not provided, prints to stdout."
)
def sketch_fits(fits_path, output):
    """
    Extract sketch statistics from a FITS file.
    
    Example:
        speedx sketch-fits observation.fits -o sketch.json
    """
    import json
    
    click.echo(f"Extracting sketch from {fits_path}...")
    
    sketch = extract_fits_sketch(fits_path)
    
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(sketch, f, indent=2)
        click.echo(f"Sketch saved to {output}")
    else:
        import json
        click.echo(json.dumps(sketch, indent=2))


if __name__ == "__main__":
    main()
