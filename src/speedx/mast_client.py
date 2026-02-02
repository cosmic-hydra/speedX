"""
MAST Client for querying HST observations and products.

Provides a minimal client to interact with the MAST API without authentication.
"""
import requests
import pandas as pd
from typing import Optional, Dict, List, Any
import time
import os
from pathlib import Path


class MASTClient:
    """
    Client for querying the MAST (Mikulski Archive for Space Telescopes) API.
    
    This client provides methods to:
    - Query HST observations by various criteria
    - Fetch metadata and save to parquet/CSV
    - Download observation products (FITS files)
    
    Parameters
    ----------
    base_url : str, optional
        Base URL for MAST API. Default is the public MAST API endpoint.
    timeout : int, optional
        Request timeout in seconds. Default is 30.
    rate_limit_delay : float, optional
        Delay between requests in seconds to respect rate limits. Default is 0.1.
    """
    
    DEFAULT_BASE_URL = "https://mast.stsci.edu/api/v0.1"
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
        rate_limit_delay: float = 0.1
    ):
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _build_query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        sort_by: Optional[List[tuple]] = None,
        page_size: int = 100,
        page: int = 1
    ) -> Dict[str, Any]:
        """
        Build a MAST API query.
        
        Parameters
        ----------
        filters : dict, optional
            Filter criteria as column:value pairs.
        columns : list of str, optional
            Columns to return. If None, returns all columns.
        sort_by : list of tuple, optional
            List of (column, direction) tuples for sorting.
        page_size : int
            Number of results per page.
        page : int
            Page number (1-indexed).
            
        Returns
        -------
        query : dict
            Query dictionary for MAST API.
        """
        query = {
            "service": "Mast.Caom.Filtered",
            "format": "json",
            "pagesize": page_size,
            "page": page
        }
        
        if filters:
            query["filters"] = [
                {
                    "paramName": key,
                    "values": [value] if not isinstance(value, list) else value
                }
                for key, value in filters.items()
            ]
        
        if columns:
            query["columns"] = ",".join(columns)
        
        if sort_by:
            query["sort_by"] = [
                {"field": col, "direction": direction}
                for col, direction in sort_by
            ]
        
        return query
    
    def query_observations(
        self,
        instrument: Optional[str] = None,
        target_name: Optional[str] = None,
        proposal_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        max_records: Optional[int] = 1000
    ) -> pd.DataFrame:
        """
        Query HST observations.
        
        Parameters
        ----------
        instrument : str, optional
            Instrument name (e.g., 'ACS/WFC', 'WFC3/UVIS').
        target_name : str, optional
            Target name to search for.
        proposal_id : str, optional
            Proposal ID to filter by.
        filters : dict, optional
            Additional filter criteria.
        columns : list of str, optional
            Columns to return.
        max_records : int, optional
            Maximum number of records to retrieve.
            
        Returns
        -------
        observations : pd.DataFrame
            DataFrame containing observation metadata.
        """
        # Build filters
        query_filters = filters.copy() if filters else {}
        
        if instrument:
            query_filters["instrument_name"] = instrument
        
        if target_name:
            query_filters["target_name"] = target_name
        
        if proposal_id:
            query_filters["proposal_id"] = proposal_id
        
        # Default columns if not specified
        if columns is None:
            columns = [
                "obs_id",
                "target_name",
                "instrument_name",
                "filters",
                "proposal_id",
                "t_exptime",
                "dataproduct_type",
                "obs_collection"
            ]
        
        # Query in pages
        all_data = []
        page = 1
        page_size = min(100, max_records) if max_records else 100
        
        while True:
            self._rate_limit()
            
            query = self._build_query(
                filters=query_filters,
                columns=columns,
                page_size=page_size,
                page=page
            )
            
            try:
                response = requests.post(
                    f"{self.base_url}/invoke",
                    json=query,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                if "data" not in data or not data["data"]:
                    break
                
                all_data.extend(data["data"])
                
                # Check if we've reached max_records or end of results
                if max_records and len(all_data) >= max_records:
                    all_data = all_data[:max_records]
                    break
                
                if len(data["data"]) < page_size:
                    break
                
                page += 1
                
            except requests.RequestException as e:
                raise RuntimeError(f"MAST API request failed: {e}")
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
        else:
            df = pd.DataFrame(columns=columns)
        
        return df
    
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
    
    def get_product_list(
        self,
        obs_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get list of data products for an observation.
        
        Parameters
        ----------
        obs_id : str
            Observation ID.
            
        Returns
        -------
        products : list of dict
            List of data products with metadata.
        """
        self._rate_limit()
        
        query = {
            "service": "Mast.Caom.Products",
            "format": "json",
            "params": {
                "obsid": obs_id
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/invoke",
                json=query,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            if "data" in data:
                return data["data"]
            else:
                return []
                
        except requests.RequestException as e:
            raise RuntimeError(f"MAST API request failed: {e}")
    
    def download_product(
        self,
        product_uri: str,
        output_dir: str,
        overwrite: bool = False
    ) -> str:
        """
        Download a data product.
        
        Parameters
        ----------
        product_uri : str
            Product URI (typically from product list).
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
        
        # Extract filename from URI
        filename = product_uri.split("/")[-1]
        output_path = output_dir / filename
        
        if output_path.exists() and not overwrite:
            return str(output_path)
        
        self._rate_limit()
        
        # Construct download URL
        download_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={product_uri}"
        
        try:
            response = requests.get(
                download_url,
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
    
    def fetch_sample(
        self,
        n_samples: int = 10,
        instrument: Optional[str] = None,
        output_dir: Optional[str] = None,
        product_type: str = "SCIENCE"
    ) -> List[str]:
        """
        Fetch a small sample of observations and download products.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to fetch. Default is 10.
        instrument : str, optional
            Instrument name to filter by.
        output_dir : str, optional
            Output directory for downloads. If None, uses './data/samples'.
        product_type : str, optional
            Type of product to download. Default is 'SCIENCE'.
            
        Returns
        -------
        downloaded_files : list of str
            List of paths to downloaded files.
        """
        if output_dir is None:
            output_dir = "./data/samples"
        
        # Query observations
        observations = self.query_observations(
            instrument=instrument,
            max_records=n_samples
        )
        
        if observations.empty:
            return []
        
        downloaded_files = []
        
        # Download first product from each observation
        for _, obs in observations.iterrows():
            obs_id = obs["obs_id"]
            
            try:
                products = self.get_product_list(obs_id)
                
                # Find a SCIENCE product
                science_products = [
                    p for p in products
                    if p.get("productType") == product_type
                ]
                
                if science_products:
                    product = science_products[0]
                    product_uri = product.get("dataURI")
                    
                    if product_uri:
                        file_path = self.download_product(
                            product_uri,
                            output_dir
                        )
                        downloaded_files.append(file_path)
                
            except Exception as e:
                # Skip failed downloads
                print(f"Warning: Failed to download products for {obs_id}: {e}")
                continue
        
        return downloaded_files
