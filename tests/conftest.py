import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

JAIPUR = (26.9124, 75.7873)


def make_clear_weather(lat: float, lon: float, start: str = "2023-01-01", days: int = 365,
                       temp: float = 30.0, wind: float = 2.0, cloud_factor: float = 1.0) -> pd.DataFrame:
    """Synthetic WeatherFrame from the Ineichen clear-sky model (UTC, interval centres, W/m²)."""
    idx = pd.date_range(start, periods=days * 24, freq="h", tz="UTC") + pd.Timedelta(minutes=30)
    cs = pvlib.location.Location(lat, lon).get_clearsky(idx, model="ineichen")
    df = pd.DataFrame({
        "ghi": cs["ghi"] * cloud_factor, "dni": cs["dni"] * cloud_factor, "dhi": cs["dhi"],
        "temp_air": temp + 5 * np.sin(2 * np.pi * (idx.hour - 3) / 24),
        "wind_speed": wind, "relative_humidity": 40.0, "cloud_cover": (1 - cloud_factor) * 100,
    }, index=idx.rename("time"))
    df.attrs.update(source="synthetic-clearsky", latitude=lat, longitude=lon, elevation=400.0)
    return df


@pytest.fixture
def jaipur_year():
    return make_clear_weather(*JAIPUR)


@pytest.fixture
def jaipur_week():
    return make_clear_weather(*JAIPUR, start="2023-03-01", days=8)
