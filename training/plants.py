"""
Registry of the real, metered PV plants in data/real_plants_hourly.csv with
the physical specs the physics model needs.

Sources: NREL PVDAQ / 2023 Solar Data Prize system metadata (cache/pvdaq/*.json,
cache/nrel_prize/*.json). AC limits come from inverter nameplates where the
metadata gives them; otherwise `ac_kw=None` and the limit is taken from the
99.9th percentile of metered output.

`kwp` is module count x module rating where the metadata lists modules
(1423: 24 x 270 W = 6.48 kWp, although its system record says 6.0).
`commissioned` lets training remove age degradation so every plant is
compared as if new.
"""

PLANTS = {
    "nrel_2107_arbuckle_ca": dict(lat=38.996306, lon=-122.134111, kwp=893.0, tilt=25.0, azimuth=180.0,
                                  ac_kw=None, tech="c-Si", climate="Csa", commissioned="2017-01-01"),
    "pvdaq_34_las_vegas_nv": dict(lat=36.1952, lon=-115.1582, kwp=146.64, tilt=11.2, azimuth=180.0,
                                  ac_kw=135.0, tech="mono-Si", climate="BWh", commissioned="2010-08-23"),
    "pvdaq_35_las_vegas_nv": dict(lat=36.1952, lon=-115.1582, kwp=121.68, tilt=None, azimuth=270.0,
                                  ac_kw=135.0, tech="mono-Si", climate="BWh", commissioned="2010-08-23"),
    "pvdaq_1423_henderson_nv": dict(lat=36.0275, lon=-114.9215, kwp=6.48, tilt=35.0, azimuth=180.0,
                                    ac_kw=None, tech="mono-Si", climate="BWh", commissioned="2016-02-01"),
    "pvdaq_10_golden_co": dict(lat=39.7404, lon=-105.1774, kwp=1.12, tilt=40.0, azimuth=180.0,
                               ac_kw=None, tech="CIS thin-film", climate="BSk", commissioned="2006-01-09"),
}

# Typical crystalline degradation; thin film ages faster.
DEGRADATION_PCT_PER_YEAR = {"c-Si": 0.5, "mono-Si": 0.5, "CIS thin-film": 1.0}
