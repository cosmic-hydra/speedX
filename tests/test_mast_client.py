"""Tests for MAST client."""
import pytest
import pandas as pd
import responses
import json
from pathlib import Path
import tempfile

from speedx.mast_client import MASTClient


@pytest.fixture
def mast_client():
    """Create a MAST client for testing."""
    return MASTClient(rate_limit_delay=0.0)  # No delay for tests


@pytest.fixture
def mock_observations_response():
    """Mock MAST API response for observations."""
    return {
        "data": [
            {
                "obs_id": "hst_12345_01_acs_wfc_f606w",
                "target_name": "NGC1234",
                "instrument_name": "ACS/WFC",
                "filters": "F606W",
                "proposal_id": "12345",
                "t_exptime": 100.0,
                "dataproduct_type": "image",
                "obs_collection": "HST"
            },
            {
                "obs_id": "hst_12345_02_acs_wfc_f814w",
                "target_name": "NGC1234",
                "instrument_name": "ACS/WFC",
                "filters": "F814W",
                "proposal_id": "12345",
                "t_exptime": 200.0,
                "dataproduct_type": "image",
                "obs_collection": "HST"
            }
        ]
    }


@pytest.fixture
def mock_products_response():
    """Mock MAST API response for products."""
    return {
        "data": [
            {
                "dataURI": "mast:HST/product/test_file.fits",
                "productType": "SCIENCE",
                "size": 1024000
            }
        ]
    }


def test_client_initialization():
    """Test MAST client initialization."""
    client = MASTClient()
    assert client.base_url == MASTClient.DEFAULT_BASE_URL
    assert client.timeout == 30
    
    client = MASTClient(base_url="http://custom.url", timeout=60)
    assert client.base_url == "http://custom.url"
    assert client.timeout == 60


def test_build_query(mast_client):
    """Test query building."""
    query = mast_client._build_query(
        filters={"instrument_name": "ACS/WFC", "proposal_id": "12345"},
        columns=["obs_id", "target_name"],
        page_size=50,
        page=2
    )
    
    assert query["service"] == "Mast.Caom.Filtered"
    assert query["format"] == "json"
    assert query["pagesize"] == 50
    assert query["page"] == 2
    assert len(query["filters"]) == 2
    assert query["columns"] == "obs_id,target_name"


def test_build_query_with_sort(mast_client):
    """Test query building with sorting."""
    query = mast_client._build_query(
        sort_by=[("t_exptime", "desc"), ("obs_id", "asc")]
    )
    
    assert "sort_by" in query
    assert len(query["sort_by"]) == 2
    assert query["sort_by"][0]["field"] == "t_exptime"
    assert query["sort_by"][0]["direction"] == "desc"


@responses.activate
def test_query_observations(mast_client, mock_observations_response):
    """Test querying observations with mocked response."""
    # Mock the API response
    responses.add(
        responses.POST,
        f"{mast_client.base_url}/invoke",
        json=mock_observations_response,
        status=200
    )
    
    # Query observations
    df = mast_client.query_observations(
        instrument="ACS/WFC",
        target_name="NGC1234",
        max_records=10
    )
    
    # Check result
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "obs_id" in df.columns
    assert "target_name" in df.columns
    assert df["instrument_name"].iloc[0] == "ACS/WFC"


@responses.activate
def test_query_observations_pagination(mast_client):
    """Test pagination in query_observations."""
    # First page
    responses.add(
        responses.POST,
        f"{mast_client.base_url}/invoke",
        json={"data": [{"obs_id": f"obs_{i}"} for i in range(100)]},
        status=200
    )
    
    # Second page (partial)
    responses.add(
        responses.POST,
        f"{mast_client.base_url}/invoke",
        json={"data": [{"obs_id": f"obs_{i}"} for i in range(100, 150)]},
        status=200
    )
    
    # Third page (empty)
    responses.add(
        responses.POST,
        f"{mast_client.base_url}/invoke",
        json={"data": []},
        status=200
    )
    
    # Query with max_records
    df = mast_client.query_observations(max_records=150)
    
    assert len(df) == 150


@responses.activate
def test_query_observations_empty(mast_client):
    """Test query with no results."""
    responses.add(
        responses.POST,
        f"{mast_client.base_url}/invoke",
        json={"data": []},
        status=200
    )
    
    df = mast_client.query_observations(instrument="NONEXISTENT")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@responses.activate
def test_query_observations_error(mast_client):
    """Test handling of API errors."""
    responses.add(
        responses.POST,
        f"{mast_client.base_url}/invoke",
        json={"error": "Internal server error"},
        status=500
    )
    
    with pytest.raises(RuntimeError, match="MAST API request failed"):
        mast_client.query_observations()


def test_save_and_load_metadata_parquet(mast_client):
    """Test saving and loading metadata in parquet format."""
    df = pd.DataFrame({
        "obs_id": ["obs_1", "obs_2"],
        "target_name": ["Target1", "Target2"],
        "instrument_name": ["ACS/WFC", "WFC3/UVIS"]
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.parquet"
        
        # Save
        mast_client.save_metadata(df, str(output_path), format="parquet")
        assert output_path.exists()
        
        # Load
        df_loaded = mast_client.load_metadata(str(output_path))
        pd.testing.assert_frame_equal(df, df_loaded)


def test_save_and_load_metadata_csv(mast_client):
    """Test saving and loading metadata in CSV format."""
    df = pd.DataFrame({
        "obs_id": ["obs_1", "obs_2"],
        "target_name": ["Target1", "Target2"]
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.csv"
        
        # Save
        mast_client.save_metadata(df, str(output_path), format="csv")
        assert output_path.exists()
        
        # Load
        df_loaded = mast_client.load_metadata(str(output_path))
        pd.testing.assert_frame_equal(df, df_loaded)


def test_save_metadata_invalid_format(mast_client):
    """Test error handling for invalid format."""
    df = pd.DataFrame({"col": [1, 2]})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.txt"
        
        with pytest.raises(ValueError, match="Unsupported format"):
            mast_client.save_metadata(df, str(output_path), format="txt")


@responses.activate
def test_get_product_list(mast_client, mock_products_response):
    """Test getting product list for an observation."""
    responses.add(
        responses.POST,
        f"{mast_client.base_url}/invoke",
        json=mock_products_response,
        status=200
    )
    
    products = mast_client.get_product_list("hst_12345_01_acs_wfc_f606w")
    
    assert len(products) == 1
    assert products[0]["dataURI"] == "mast:HST/product/test_file.fits"
    assert products[0]["productType"] == "SCIENCE"


@responses.activate
def test_get_product_list_empty(mast_client):
    """Test getting product list with no results."""
    responses.add(
        responses.POST,
        f"{mast_client.base_url}/invoke",
        json={"data": []},
        status=200
    )
    
    products = mast_client.get_product_list("nonexistent_obs")
    assert len(products) == 0


@responses.activate
def test_download_product(mast_client):
    """Test downloading a product."""
    # Mock download response
    responses.add(
        responses.GET,
        "https://mast.stsci.edu/api/v0.1/Download/file",
        body=b"FITS data here",
        status=200
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = mast_client.download_product(
            "mast:HST/product/test.fits",
            tmpdir
        )
        
        assert Path(file_path).exists()
        with open(file_path, "rb") as f:
            content = f.read()
        assert content == b"FITS data here"


@responses.activate
def test_download_product_no_overwrite(mast_client):
    """Test that download respects overwrite flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.fits"
        
        # Create existing file
        output_path.write_text("existing content")
        
        # Download without overwrite
        file_path = mast_client.download_product(
            "mast:HST/product/test.fits",
            tmpdir,
            overwrite=False
        )
        
        # Should return existing file without downloading
        assert file_path == str(output_path)
        assert output_path.read_text() == "existing content"


def test_rate_limiting(mast_client):
    """Test that rate limiting is applied."""
    import time
    
    client = MASTClient(rate_limit_delay=0.1)
    
    start = time.time()
    client._rate_limit()
    client._rate_limit()
    elapsed = time.time() - start
    
    # Should have at least one delay
    assert elapsed >= 0.1
