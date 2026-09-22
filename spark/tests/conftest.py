import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep import get_spark  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    session = get_spark(app_name="pytest-ecommerce-bigdata")
    yield session
    session.stop()
