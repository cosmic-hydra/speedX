"""Tests for IRSA client."""
import pytest
import pandas as pd
import responses
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock

from speedx.irsa_client import IRSAClient


@pytest.fixture
def irsa_client():
    """Create an IRSA client for testing."""
    return IRSAClient(rate_limit_delay=0.0)  # No delay for tests


@pytest.fixture
def mock_votable_response():
    """Mock IRSA VOTable response."""
    # Simple VOTable XML format
    votable_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">
  <RESOURCE type="results">
    <TABLE>
      <FIELD name="ra" datatype="double" unit="deg"/>
      <FIELD name="dec" datatype="double" unit="deg"/>
      <FIELD name="scan_id" datatype="char" arraysize="*"/>
      <FIELD name="frame_num" datatype="int"/>
      <FIELD name="w1mpro" datatype="float"/>
      <DATA>
        <TABLEDATA>
          <TR>
            <TD>83.6333</TD>
            <TD>22.0144</TD>
            <TD>00001a001</TD>
            <TD>1</TD>
            <TD>12.5</TD>
          </TR>
          <TR>
            <TD>83.6334</TD>
            <TD>22.0145</TD>
            <TD>00001a001</TD>
            <TD>2</TD>
            <TD>13.2</TD>
          </TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>"""
    return votable_xml


def test_client_initialization():
    """Test IRSA client initialization."""
    client = IRSAClient()
    assert client.base_url == IRSAClient.DEFAULT_BASE_URL
    assert client.timeout == 30
    
    client = IRSAClient(base_url="http://custom.url", timeout=60)
    assert client.base_url == "http://custom.url"
    assert client.timeout == 60


@responses.activate
def test_query_neowise(irsa_client, mock_votable_response):
    """Test querying NEOWISE observations with mocked response."""
    # Mock the API response
    responses.add(
        responses.GET,
        f"{irsa_client.base_url}/wise/neowiser/p1bm_frm",
        body=mock_votable_response,
        status=200
    )
    
    # Query observations
    df = irsa_client.query_neowise(
        ra=83.6333,
        dec=22.0144,
        size=0.1,
        mission="wise",
        dataset="neowiser",
        table="p1bm_frm"
    )
    
    # Check result
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "ra" in df.columns
    assert "dec" in df.columns


@responses.activate
def test_query_by_position(irsa_client, mock_votable_response):
    """Test querying by position."""
    responses.add(
        responses.GET,
        f"{irsa_client.base_url}/wise/neowiser/p1bm_frm",
        body=mock_votable_response,
        status=200
    )
    
    df = irsa_client.query_by_position(
        ra=83.6333,
        dec=22.0144,
        radius=0.1
    )
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


@responses.activate
def test_query_with_max_records(irsa_client, mock_votable_response):
    """Test query with max_records parameter."""
    responses.add(
        responses.GET,
        f"{irsa_client.base_url}/wise/neowiser/p1bm_frm",
        body=mock_votable_response,
        status=200
    )
    
    df = irsa_client.query_neowise(
        ra=83.6333,
        dec=22.0144,
        size=0.1,
        max_records=100
    )
    
    # Check that MAXREC parameter was sent
    assert len(responses.calls) == 1
    assert "MAXREC=100" in responses.calls[0].request.url


@responses.activate
def test_query_error(irsa_client):
    """Test handling of API errors."""
    responses.add(
        responses.GET,
        f"{irsa_client.base_url}/wise/neowiser/p1bm_frm",
        json={"error": "Internal server error"},
        status=500
    )
    
    with pytest.raises(RuntimeError, match="IRSA API request failed"):
        irsa_client.query_neowise(ra=83.6333, dec=22.0144, size=0.1)


def test_save_and_load_metadata_parquet(irsa_client):
    """Test saving and loading metadata in parquet format."""
    df = pd.DataFrame({
        "ra": [83.6333, 83.6334],
        "dec": [22.0144, 22.0145],
        "scan_id": ["00001a001", "00001a001"]
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.parquet"
        
        # Save
        irsa_client.save_metadata(df, str(output_path), format="parquet")
        assert output_path.exists()
        
        # Load
        df_loaded = irsa_client.load_metadata(str(output_path))
        pd.testing.assert_frame_equal(df, df_loaded)


def test_save_and_load_metadata_csv(irsa_client):
    """Test saving and loading metadata in CSV format."""
    df = pd.DataFrame({
        "ra": [83.6333, 83.6334],
        "dec": [22.0144, 22.0145]
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.csv"
        
        # Save
        irsa_client.save_metadata(df, str(output_path), format="csv")
        assert output_path.exists()
        
        # Load
        df_loaded = irsa_client.load_metadata(str(output_path))
        pd.testing.assert_frame_equal(df, df_loaded)


def test_save_metadata_invalid_format(irsa_client):
    """Test error handling for invalid format."""
    df = pd.DataFrame({"col": [1, 2]})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.txt"
        
        with pytest.raises(ValueError, match="Unsupported format"):
            irsa_client.save_metadata(df, str(output_path), format="txt")


@responses.activate
def test_download_product(irsa_client):
    """Test downloading a product."""
    # Mock download response
    responses.add(
        responses.GET,
        "https://irsa.ipac.caltech.edu/ibe/data/wise/neowiser/p1bm_frm/test.fits",
        body=b"FITS data here",
        status=200
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = irsa_client.download_product(
            "https://irsa.ipac.caltech.edu/ibe/data/wise/neowiser/p1bm_frm/test.fits",
            tmpdir
        )
        
        assert Path(file_path).exists()
        with open(file_path, "rb") as f:
            content = f.read()
        assert content == b"FITS data here"


@responses.activate
def test_download_product_no_overwrite(irsa_client):
    """Test that download respects overwrite flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.fits"
        
        # Create existing file
        output_path.write_text("existing content")
        
        # Download without overwrite
        file_path = irsa_client.download_product(
            "https://irsa.ipac.caltech.edu/ibe/data/wise/neowiser/p1bm_frm/test.fits",
            tmpdir,
            overwrite=False
        )
        
        # Should return existing file without downloading
        assert file_path == str(output_path)
        assert output_path.read_text() == "existing content"


def test_rate_limiting(irsa_client):
    """Test that rate limiting is applied."""
    import time
    
    client = IRSAClient(rate_limit_delay=0.1)
    
    start = time.time()
    client._rate_limit()
    client._rate_limit()
    elapsed = time.time() - start
    
    # Should have at least one delay
    assert elapsed >= 0.1
