"""NOAA weather provider for the SoCal MIDAS scenario.

This module replaces the bundled DWD *Bremen* weather data with **NOAA**
observations for Southern California.  It produces a CSV file in *exactly* the
schema expected by ``midas_weather.model.provider.WeatherData`` so that the
existing MIDAS ``weather`` simulator can consume it unchanged -- i.e. NOAA is
wired in *inside the MIDAS scenario*, not as a separate mosaik simulator.

The MIDAS weather provider expects an hourly CSV with:

* index column 0: datetime in German format ``%d.%m.%Y %H:%M:%S`` (UTC values)
* columns (``midas_weather.meta``):
    - ``t_air_deg_celsius``          air temperature                 [degC]
    - ``day_avg_t_air_deg_celsius``  daily mean air temperature      [degC]
    - ``gh_w_per_m2``                global horizontal irradiance    [W/m2]
    - ``dh_w_per_m2``                diffuse horizontal irradiance   [W/m2]
    - ``wind_v_m_per_s``             wind speed                      [m/s]
    - ``wind_dir_deg``               wind direction                  [deg]
    - ``air_pressure_hpa``           air pressure                    [hPa]
    - ``sun_hours_min_per_h``        sunshine minutes per hour       [min/h]
    - ``cloud_percent``              cloud cover                     [%]

Two NOAA data sources are supported:

``isd`` (default, lightweight, CI-friendly)
    NCEI **Integrated Surface Database** Access Data Service.  Pure CSV, no
    API key, no GRIB parsing.  One representative SoCal station (KLAX by
    default).  Solar irradiance is reconstructed from a clear-sky model
    modulated by the observed cloud cover, because ISD does not report
    radiation.

``hrrr`` (high fidelity, heavy)
    NOAA **High-Resolution Rapid Refresh** 3 km hourly analysis from the
    public AWS bucket ``noaa-hrrr-bdp-pds`` (no-sign-request).  Requires
    ``cfgrib``/``herbie``; used for realistic Santa-Ana wind fields.

If neither source is reachable (e.g. fully offline CI), a deterministic
**synthetic Santa-Ana** fallback is generated so the pipeline never breaks.

The GER datetime format string is imported from MIDAS when available so the
output always matches whatever the installed MIDAS version expects.
"""

from __future__ import annotations

import argparse
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

try:  # keep in lock-step with the installed MIDAS
    from midas.util.dateformat import GER as _GER
except Exception:  # pragma: no cover - MIDAS not importable at import time
    _GER = "%d.%m.%Y %H:%M:%S"

LOG = logging.getLogger("noaa_provider")

# --- MIDAS weather CSV column names (mirrors midas_weather.meta) ----------
T_AIR = "t_air_deg_celsius"
AVG_T_AIR = "day_avg_t_air_deg_celsius"
GHI = "gh_w_per_m2"
DHI = "dh_w_per_m2"
WIND = "wind_v_m_per_s"
WINDDIR = "wind_dir_deg"
PRESSURE = "air_pressure_hpa"
SUN_HOURS = "sun_hours_min_per_h"
CLOUD = "cloud_percent"

MIDAS_COLUMNS = [
    T_AIR, AVG_T_AIR, GHI, DHI, WIND, WINDDIR, PRESSURE, SUN_HOURS, CLOUD,
]

# Default representative SoCal stations (NCEI ISD ids = USAF+WBAN).
#   KLAX  Los Angeles Intl    72295023174
#   KSAN  San Diego Lindbergh 72290023188
#   KONT  Ontario Intl        72288703102  (inland / Santa-Ana corridor)
DEFAULT_STATION = "72295023174"  # KLAX
DEFAULT_LAT = 33.94
DEFAULT_LON = -118.41

ISD_ACCESS_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


# --------------------------------------------------------------------------
# clear-sky solar model (used to reconstruct GHI/DHI for ISD which lacks it)
# --------------------------------------------------------------------------
def _solar_position(dt_utc: datetime, lat: float, lon: float) -> float:
    """Return solar elevation angle [deg] for a UTC datetime and location."""
    doy = dt_utc.timetuple().tm_yday
    frac_hour = dt_utc.hour + dt_utc.minute / 60.0
    decl = 23.45 * math.sin(math.radians(360.0 * (284 + doy) / 365.0))
    # equation of time (minutes), Spencer approximation simplified
    b = math.radians(360.0 * (doy - 81) / 364.0)
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    solar_time = frac_hour + (4 * lon + eot) / 60.0
    hour_angle = 15.0 * (solar_time - 12.0)
    lat_r = math.radians(lat)
    decl_r = math.radians(decl)
    ha_r = math.radians(hour_angle)
    sin_elev = (
        math.sin(lat_r) * math.sin(decl_r)
        + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r)
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))


def _clear_sky_ghi(elev_deg: float) -> float:
    """Simple Haurwitz clear-sky GHI model [W/m2] from solar elevation."""
    if elev_deg <= 0:
        return 0.0
    zenith = math.radians(90.0 - elev_deg)
    cos_z = math.cos(zenith)
    return max(0.0, 1098.0 * cos_z * math.exp(-0.059 / max(cos_z, 1e-3)))


def _ghi_dhi_from_cloud(dt_utc, lat, lon, cloud_pct):
    """Reconstruct (GHI, DHI) from clear-sky model attenuated by cloud."""
    elev = _solar_position(dt_utc, lat, lon)
    cs_ghi = _clear_sky_ghi(elev)
    c = max(0.0, min(1.0, cloud_pct / 100.0))
    # Kasten-Czeplak cloud attenuation of GHI
    ghi = cs_ghi * (1.0 - 0.75 * c ** 3.4)
    # diffuse fraction grows with cloudiness
    diff_frac = min(1.0, 0.15 + 0.85 * c)
    dhi = ghi * diff_frac
    return max(0.0, ghi), max(0.0, dhi)


# --------------------------------------------------------------------------
# ISD parsing
# --------------------------------------------------------------------------
def _parse_isd_wnd(field_str: str):
    """ISD WND field: dir,dirQC,type,speed,speedQC. speed is m/s x10."""
    try:
        parts = field_str.split(",")
        direction = float(parts[0])
        speed = float(parts[3]) / 10.0
        if direction >= 999:
            direction = np.nan
        if speed >= 999:
            speed = np.nan
        return direction, speed
    except Exception:
        return np.nan, np.nan


def _parse_isd_tmp(field_str: str):
    """ISD TMP field: value,QC. value is degC x10 (signed)."""
    try:
        val = float(field_str.split(",")[0]) / 10.0
        return np.nan if abs(val) >= 999 else val
    except Exception:
        return np.nan


def _parse_isd_slp(field_str: str):
    """ISD SLP field: value,QC. value is hPa x10."""
    try:
        val = float(field_str.split(",")[0]) / 10.0
        return np.nan if val >= 9999 else val
    except Exception:
        return np.nan


def _parse_isd_cloud(row) -> float:
    """Best-effort cloud cover [%] from GA1/GF1 sky-cover oktas if present."""
    for col in ("GF1", "GA1"):
        if col in row and isinstance(row[col], str) and row[col]:
            try:
                oktas = float(row[col].split(",")[0])
                if oktas <= 8:
                    return 100.0 * oktas / 8.0
            except Exception:
                pass
    return np.nan


def fetch_isd(
    station: str,
    start: datetime,
    end: datetime,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch NOAA NCEI ISD global-hourly observations and map to MIDAS schema."""
    import requests

    params = {
        "dataset": "global-hourly",
        "stations": station,
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "format": "csv",
        "dataTypes": "TMP,WND,SLP,GA1,GF1",
    }
    LOG.info("Fetching NOAA ISD station %s %s..%s", station, params["startDate"], params["endDate"])
    resp = requests.get(ISD_ACCESS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), dtype=str)
    if raw.empty:
        raise RuntimeError("NOAA ISD returned no rows")

    raw["DATE"] = pd.to_datetime(raw["DATE"], utc=True)
    raw = raw.set_index("DATE").sort_index()

    t_air, wind_v, wind_d, slp, cloud = [], [], [], [], []
    for _, row in raw.iterrows():
        t_air.append(_parse_isd_tmp(row.get("TMP", "")))
        d, s = _parse_isd_wnd(row.get("WND", ""))
        wind_d.append(d)
        wind_v.append(s)
        slp.append(_parse_isd_slp(row.get("SLP", "")))
        cloud.append(_parse_isd_cloud(row))

    df = pd.DataFrame(
        {
            T_AIR: t_air,
            WIND: wind_v,
            WINDDIR: wind_d,
            PRESSURE: slp,
            CLOUD: cloud,
        },
        index=raw.index,
    )
    # resample to clean hourly grid and interpolate gaps
    df = df.resample("1h").mean(numeric_only=True)
    df = df.interpolate(limit_direction="both")
    df[CLOUD] = df[CLOUD].fillna(0.0).clip(0, 100)
    df[PRESSURE] = df[PRESSURE].fillna(1013.0)
    return _finalize(df, lat, lon)


# --------------------------------------------------------------------------
# HRRR (high fidelity wind) -- optional heavy path
# --------------------------------------------------------------------------
def fetch_hrrr(
    start: datetime,
    end: datetime,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """Fetch NOAA HRRR 3 km hourly analysis (wind/temp) via Herbie/AWS.

    Requires the optional ``herbie-data`` + ``cfgrib`` extras.  Falls back to
    raising so the caller can choose the ISD path.
    """
    from herbie import Herbie  # optional dep

    rows = []
    hour = start.replace(minute=0, second=0, microsecond=0)
    while hour <= end:
        H = Herbie(hour.strftime("%Y-%m-%d %H:00"), model="hrrr", product="sfc", fxx=0)
        ds = H.xarray(":(?:TMP:2 m|UGRD:10 m|VGRD:10 m|PRES:surface|TCDC):")
        # nearest grid point to (lat, lon)
        abslat = np.abs(ds.latitude - lat)
        abslon = np.abs(ds.longitude - (lon % 360))
        dist = abslat ** 2 + abslon ** 2
        yx = np.unravel_index(int(dist.argmin()), dist.shape)
        u = float(ds["u10"].values[yx]) if "u10" in ds else float(ds["UGRD"].values[yx])
        v = float(ds["v10"].values[yx]) if "v10" in ds else float(ds["VGRD"].values[yx])
        t2 = float(ds["t2m"].values[yx]) - 273.15
        speed = math.hypot(u, v)
        direction = (math.degrees(math.atan2(-u, -v))) % 360.0
        rows.append(
            {
                "DATE": hour.replace(tzinfo=timezone.utc),
                T_AIR: t2,
                WIND: speed,
                WINDDIR: direction,
                PRESSURE: 1013.0,
                CLOUD: 0.0,
            }
        )
        hour += timedelta(hours=1)
    df = pd.DataFrame(rows).set_index("DATE")
    return _finalize(df, lat, lon)


# --------------------------------------------------------------------------
# synthetic Santa-Ana fallback (deterministic, offline-safe)
# --------------------------------------------------------------------------
def synth_santa_ana(
    start: datetime,
    end: datetime,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    seed: int = 42,
    peak_wind_ms: float = 22.0,
) -> pd.DataFrame:
    """Deterministic Santa-Ana style hourly weather (offline fallback).

    Dry, hot, strong NE (offshore) winds with a diurnal peak around midnight
    to early morning, mimicking a Santa-Ana wind event.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        start.replace(minute=0, second=0, microsecond=0),
        end.replace(minute=0, second=0, microsecond=0),
        freq="1h",
        tz="UTC",
    )
    hours = np.array([t.hour for t in idx])
    # Santa-Ana winds peak overnight/early morning (local), trough afternoon
    diurnal = 0.5 * (1 + np.cos(2 * np.pi * (hours - 6) / 24.0))
    wind = peak_wind_ms * (0.55 + 0.45 * diurnal) + rng.normal(0, 1.2, len(idx))
    wind = np.clip(wind, 0, None)
    # offshore wind from the NE (Santa-Ana): ~45 deg +/- jitter
    wdir = (45.0 + rng.normal(0, 12, len(idx))) % 360.0
    # hot & dry, diurnal temperature
    t_air = 26.0 + 9.0 * np.sin(2 * np.pi * (hours - 9) / 24.0) + rng.normal(0, 0.6, len(idx))
    cloud = np.clip(rng.normal(4, 3, len(idx)), 0, 100)  # very low cloud
    press = 1018.0 + rng.normal(0, 1.0, len(idx))  # high pressure offshore
    df = pd.DataFrame(
        {T_AIR: t_air, WIND: wind, WINDDIR: wdir, PRESSURE: press, CLOUD: cloud},
        index=idx,
    )
    return _finalize(df, lat, lon)


# --------------------------------------------------------------------------
# finalize: add solar columns, daily-avg temp, sun hours; order columns
# --------------------------------------------------------------------------
def _finalize(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    ghi, dhi, sun_h = [], [], []
    for ts, row in df.iterrows():
        dt_utc = ts.to_pydatetime().astimezone(timezone.utc)
        g, d = _ghi_dhi_from_cloud(dt_utc, lat, lon, float(row.get(CLOUD, 0.0)))
        ghi.append(g)
        dhi.append(d)
        # sunshine minutes per hour ~ proportional to clear-sky fraction
        cs = _clear_sky_ghi(_solar_position(dt_utc, lat, lon))
        sun_h.append(60.0 * (g / cs) if cs > 5 else 0.0)
    df[GHI] = ghi
    df[DHI] = dhi
    df[SUN_HOURS] = np.clip(sun_h, 0, 60)

    # daily mean air temperature
    daily = df[T_AIR].resample("1D").transform("mean")
    df[AVG_T_AIR] = daily

    for col in MIDAS_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[MIDAS_COLUMNS]
    df = df.interpolate(limit_direction="both").ffill().bfill()
    return df


def write_midas_csv(df: pd.DataFrame, path: str) -> str:
    """Write the dataframe to the MIDAS weather CSV schema (GER datetime)."""
    out = df.copy()
    out.index = [ts.strftime(_GER) for ts in pd.to_datetime(out.index, utc=True)]
    out.index.name = "Datetime"
    out.to_csv(path)
    LOG.info("Wrote NOAA weather CSV: %s (%d rows)", path, len(out))
    return path


@dataclass
class NOAAWeatherConfig:
    source: str = "isd"            # isd | hrrr | synthetic
    station: str = DEFAULT_STATION
    lat: float = DEFAULT_LAT
    lon: float = DEFAULT_LON
    start: datetime = field(default_factory=lambda: datetime(2024, 7, 16, tzinfo=timezone.utc))
    days: int = 1
    allow_fallback: bool = True


def build_noaa_weather(cfg: NOAAWeatherConfig, out_path: str) -> str:
    """Build a MIDAS-ready NOAA weather CSV.  Returns the output path.

    Pads the requested window by one day on each side so MIDAS interpolation
    and the daily-average never run off the edge of the data.
    """
    start = cfg.start - timedelta(days=1)
    end = cfg.start + timedelta(days=cfg.days + 1)

    df: Optional[pd.DataFrame] = None
    if cfg.source == "isd":
        try:
            df = fetch_isd(cfg.station, start, end, cfg.lat, cfg.lon)
        except Exception as exc:  # pragma: no cover - network dependent
            LOG.warning("ISD fetch failed (%s)", exc)
    elif cfg.source == "hrrr":
        try:
            df = fetch_hrrr(start, end, cfg.lat, cfg.lon)
        except Exception as exc:  # pragma: no cover - heavy/optional
            LOG.warning("HRRR fetch failed (%s)", exc)
    elif cfg.source == "synthetic":
        df = synth_santa_ana(start, end, cfg.lat, cfg.lon)

    if df is None or df.empty:
        if not cfg.allow_fallback:
            raise RuntimeError(f"NOAA source '{cfg.source}' unavailable and fallback disabled")
        LOG.warning("Falling back to deterministic synthetic Santa-Ana weather")
        df = synth_santa_ana(start, end, cfg.lat, cfg.lon)

    return write_midas_csv(df, out_path)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build a NOAA weather CSV for the SoCal MIDAS scenario")
    p.add_argument("--source", choices=["isd", "hrrr", "synthetic"], default="isd")
    p.add_argument("--station", default=DEFAULT_STATION)
    p.add_argument("--lat", type=float, default=DEFAULT_LAT)
    p.add_argument("--lon", type=float, default=DEFAULT_LON)
    p.add_argument("--start", default="2024-07-16", help="YYYY-MM-DD (UTC)")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--no-fallback", action="store_true")
    args = p.parse_args(argv)

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cfg = NOAAWeatherConfig(
        source=args.source,
        station=args.station,
        lat=args.lat,
        lon=args.lon,
        start=start,
        days=args.days,
        allow_fallback=not args.no_fallback,
    )
    path = build_noaa_weather(cfg, args.out)
    print(path)


if __name__ == "__main__":
    main()
