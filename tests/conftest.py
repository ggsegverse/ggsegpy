from __future__ import annotations

import pandas as pd
import pytest

from ggsegpy import dk, aseg, tracula


@pytest.fixture
def dk_atlas():
    return dk()


@pytest.fixture
def aseg_atlas():
    return aseg()


@pytest.fixture
def tracula_atlas():
    return tracula()


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "label": ["lh_bankssts", "rh_bankssts", "lh_cuneus"],
        "value": [0.5, 0.8, 0.3],
    })


@pytest.fixture
def region_data():
    return pd.DataFrame({
        "region": ["bankssts", "cuneus", "fusiform"],
        "value": [0.5, 0.8, 0.3],
    })
