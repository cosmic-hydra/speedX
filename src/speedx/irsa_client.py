"""
IRSA Client for querying NEOWISE mission data.

Provides a minimal client to interact with the IRSA SIA (Simple Image Access) API.
"""
import requests
import pandas as pd
from typing import Optional, Dict, List, Any
import time
from pathlib import Path
from astropy.io.votable import parse_single_table
from io import BytesIO


class IRSAClient:
    """
    Client for querying the IRSA (Infrared Science Archive) NEOWISE data.
    
    This client provides methods to:
    - Query NEOWISE observations using SIA (Simple Image Access) protocol
    - Fetch metadata and save to parquet/CSV
    - Download observation products
    
    Parameters
    ----------
    base_url : str, optional
        Base URL for IRSA IBE SIA API. Default is the public IRSA endpoint.
    timeout : int, optional
        Request timeout in seconds. Default is 30.
    rate_limit_delay : float, optional
        Delay between requests in seconds to respect rate limits. Default is 0.1.
    max_retries : int, optional
        Maximum number of retries for failed requests. Default is 3.
    """
    
    DEFAULT_BASE_URL = "https://irsa.ipac.caltech.edu/ibe/sia"
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
        rate_limit_delay: float = 0.1,
        max_retries: int = 3
    ):
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def query_neowise(
        self,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        size: Optional[float] = None,
        mission: str = "wise",
        dataset: str = "neowiser",
        table: str = "p1bm_frm",
        max_records: Optional[int] = None,
        **query_params
    ) -> pd.DataFrame:
        """
        Query NEOWISE observations using SIA protocol.
        
        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (J2000).
        dec : float, optional
            Declination in degrees (J2000).
        size : float, optional
            Search radius in degrees.
        mission : str, optional
            Mission name. Default is 'wise'.
        dataset : str, optional
            Dataset name. Default is 'neowiser'.
        table : str, optional
            Table name. Default is 'p1bm_frm'.
        max_records : int, optional
            Maximum number of records to retrieve.
        **query_params : dict
            Additional query parameters to pass to the SIA service.
            
        Returns
        -------
        observations : pd.DataFrame
            DataFrame containing observation metadata.
            
        Notes
        -----
        The SIA protocol expects queries in the format:
        {base_url}/{mission}/{dataset}/{table}?POS={ra},{dec}&SIZE={size}
        
        Common query parameters:
        - POS: Position as "ra,dec" in degrees
        - SIZE: Search radius in degrees
        - BAND: Spectral band filter
        - TIME: Time range
        - FORMAT: Output format (default is VOTable)
        """
        self._rate_limit()
        
        # Construct the SIA endpoint
        sia_url = f"{self.base_url}/{mission}/{dataset}/{table}"
        
        # Build query parameters
        params = {}
        
        if ra is not None and dec is not None:
            params["POS"] = f"{ra},{dec}"
        
        if size is not None:
            params["SIZE"] = size
        
        if max_records is not None:
            params["MAXREC"] = max_records
        
        # Add any additional query parameters
        params.update(query_params)
        
        # Retry logic for failed requests
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    sia_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Parse VOTable response
                votable = parse_single_table(BytesIO(response.content))
                df = votable.to_table().to_pandas()
                
                break  # Success
                
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * self.rate_limit_delay
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        f"IRSA API request failed after {self.max_retries} attempts: {e}"
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to parse IRSA response: {e}")
        
        return df
    
    def query_by_position(
        self,
        ra: float,
        dec: float,
        radius: float = 0.1,
        mission: str = "wise",
        dataset: str = "neowiser",
        table: str = "p1bm_frm",
        max_records: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query NEOWISE observations by sky position.
        
        Parameters
        ----------
        ra : float
            Right Ascension in degrees (J2000).
        dec : float
            Declination in degrees (J2000).
        radius : float, optional
            Search radius in degrees. Default is 0.1 degrees.
        mission : str, optional
            Mission name. Default is 'wise'.
        dataset : str, optional
            Dataset name. Default is 'neowiser'.
        table : str, optional
            Table name. Default is 'p1bm_frm'.
        max_records : int, optional
            Maximum number of records to retrieve.
            
        Returns
        -------
        observations : pd.DataFrame
            DataFrame containing observation metadata.
        """
        return self.query_neowise(
            ra=ra,
            dec=dec,
            size=radius,
            mission=mission,
            dataset=dataset,
            table=table,
            max_records=max_records
        )
    
    def save_metadata(
        self,
        observations: pd.DataFrame,
        output_path: str,
        format: str = "parquet"
    ):
        """
        Save observation metadata to file.
        
        Parameters
        ----------
        observations : pd.DataFrame
            Observation metadata DataFrame.
        output_path : str
            Output file path.
        format : str, optional
            Output format ('parquet' or 'csv'). Default is 'parquet'.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "parquet":
            observations.to_parquet(output_path, index=False)
        elif format.lower() == "csv":
            observations.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_metadata(
        self,
        input_path: str,
        format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load observation metadata from file.
        
        Parameters
        ----------
        input_path : str
            Input file path.
        format : str, optional
            Input format. If None, inferred from file extension.
            
        Returns
        -------
        observations : pd.DataFrame
            Observation metadata DataFrame.
        """
        input_path = Path(input_path)
        
        if format is None:
            format = input_path.suffix.lstrip(".")
        
        if format.lower() == "parquet":
            return pd.read_parquet(input_path)
        elif format.lower() == "csv":
            return pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def download_product(
        self,
        access_url: str,
        output_dir: str,
        overwrite: bool = False
    ) -> str:
        """
        Download a data product.
        
        Parameters
        ----------
        access_url : str
            Product access URL (typically from query results).
        output_dir : str
            Output directory for downloaded file.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is False.
            
        Returns
        -------
        output_path : str
            Path to downloaded file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract filename from URL
        filename = access_url.split("/")[-1].split("?")[0]
        output_path = output_dir / filename
        
        if output_path.exists() and not overwrite:
            return str(output_path)
        
        self._rate_limit()
        
        try:
            response = requests.get(
                access_url,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return str(output_path)
            
        except requests.RequestException as e:
            raise RuntimeError(f"Download failed: {e}")
