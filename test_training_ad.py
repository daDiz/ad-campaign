import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ad_campaign_train import train_model


@pytest.fixture
def dummy_data():
    # Prepare dummy data for testing
    data = {
        "EMAIL": [1, 2, 3],
        "SEARCH_ENGINE": [1, 2, 3],
        "SOCIAL_MEDIA": [1, 2, 3],
        "VIDEO": [1, 2, 3],
        "REVENUE": [10.5, 9.8, 3.3],
    }
    return pd.DataFrame(data)


def test_train_model(dummy_data):
    df = dummy_data
    model, _, _, _, _ = train_model(df)
    assert isinstance(model, Pipeline)
