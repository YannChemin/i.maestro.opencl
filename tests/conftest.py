# SPDX-License-Identifier: Unlicense
"""pytest fixtures for i.maestro.opencl.

The tests need a MAESTRO checkpoint and an OpenCL device.
They are skipped unless the checkpoint path is given in I_MAESTRO_OPENCL_CHECKPOINT.
I_MAESTRO_OPENCL_PLATFORM optionally selects the OpenCL platform.

The tests create a temporary GRASS location with sample data.
"""

import os
from pathlib import Path

import grass.script as gs
import pytest
from grass.tools import Tools

# Sample data configuration
# For testing, we create small synthetic raster maps for each modality
SAMPLE_MODALITIES = {
    "aerial": {"bands": 4, "size": 64, "norm_fac": 255.0, "log_scale": False},
    "spot": {"bands": 3, "size": 64, "norm_fac": 255.0, "log_scale": False},
    "s2": {"bands": 10, "size": 16, "norm_fac": 5000.0, "log_scale": False},
    "s1": {"bands": 2, "size": 16, "norm_fac": 20.0, "log_scale": False},
    "dem": {"bands": 2, "size": 64, "norm_fac": 1000.0, "log_scale": False, "rescale_elev": True},
}


def create_sample_raster(tools, name, size, value=128):
    """Create a simple flat raster map for testing."""
    expression = f"{name} = {value}"
    tools.r_mapcalc(expression=expression)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Session with sample raster maps for MAESTRO testing.
    
    Creates a temporary GRASS location with synthetic data for testing
    the i.maestro.opencl module.
    """
    project = tmp_path_factory.mktemp("i_maestro_opencl") / "project"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        
        # Set region to accommodate sample data
        tools.g_region(n=1000, s=0, e=1000, w=0, res=1)
        
        # Create sample raster maps for each modality
        # For simplicity, create one band per modality
        # In real tests, you would create all bands
        create_sample_raster(tools, "aerial_b0", 1000)
        create_sample_raster(tools, "aerial_b1", 1000)
        create_sample_raster(tools, "aerial_b2", 1000)
        create_sample_raster(tools, "aerial_b3", 1000)
        
        create_sample_raster(tools, "spot_b0", 1000)
        create_sample_raster(tools, "spot_b1", 1000)
        create_sample_raster(tools, "spot_b2", 1000)
        
        # Sentinel-2 bands
        for i in range(10):
            create_sample_raster(tools, f"s2_b{i+1}", 1000, value=100)
        
        # Sentinel-1 bands
        create_sample_raster(tools, "s1_vv", 1000, value=5)
        create_sample_raster(tools, "s1_vh", 1000, value=3)
        
        # DEM
        create_sample_raster(tools, "dem_elev", 1000, value=200)
        create_sample_raster(tools, "dem_surface", 1000, value=180)
        
        yield session


@pytest.fixture(scope="module")
def checkpoint():
    """Get checkpoint path from environment variable."""
    path = os.environ.get("I_MAESTRO_OPENCL_CHECKPOINT")
    if not path or not Path(path).is_file():
        pytest.skip("set I_MAESTRO_OPENCL_CHECKPOINT to a MAESTRO .ckpt checkpoint")
    return path


@pytest.fixture(scope="module")
def checkpoint_variant():
    """Get checkpoint variant from environment variable."""
    variant = os.environ.get("I_MAESTRO_OPENCL_VARIANT", "flair-hub")
    return variant


@pytest.fixture(scope="module")
def device_options():
    """Get device options from environment variables."""
    platform = os.environ.get("I_MAESTRO_OPENCL_PLATFORM")
    device = os.environ.get("I_MAESTRO_OPENCL_DEVICE")
    profile = os.environ.get("I_MAESTRO_OPENCL_PROFILE")
    precision = os.environ.get("I_MAESTRO_OPENCL_PRECISION")
    
    options = {}
    if platform:
        options["platform"] = platform
    if device:
        options["device"] = device
    if profile:
        options["profile"] = profile
    if precision:
        options["precision"] = precision
    return options
