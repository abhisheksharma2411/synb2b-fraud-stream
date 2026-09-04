import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest

from t13.simulate import prepare_stream


@pytest.fixture(scope="session")
def ctx_small():
    return prepare_stream("synb2b", seed=11, small=True)


@pytest.fixture(scope="session")
def ctx_full():
    return prepare_stream("synb2b", seed=11, small=False)
