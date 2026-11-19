# Tropical nights and insufficient rest from sleep in Japan

Data and code for: Matsushita, D. "Tropical nights and insufficient rest from sleep: a nationwide panel study of 335 secondary medical care areas in Japan, 2018–2023" (submitted to *Environment International*).

## Data (`data/`)

| File | Content |
|---|---|
| `panel_area_year.csv` | Analysis dataset: 335 secondary medical care areas × fiscal years 2018–2023 (2,010 rows). |
| `ndb_q20_area_sex_age_FY2018-2023.csv` | Responses to question 20 (“Do you get enough rest from sleep?”) by area, sex, and 5-year age group, from NDB Open Data (6th–11th releases). Used for standardization and the aggregated binomial model. |
| `validation_station_year.csv` | Exposure validation at 376 air quality monitoring stations (1,996 station-years): observed and estimated tropical nights. |
| `scenario_2023_by_area.csv` | 2023 scenario by area: difference in tropical nights from the 2018–2022 mean and predicted difference in reports of insufficient rest from sleep. |

### Columns of `panel_area_year.csv`

- `area_code`, `pref_code`, `region`, `fiscal_year` (April–March)
- `responses`: number of checkup responses to question 20
- `prop_insufficient_rest_std`: proportion answering “no”, standardized to the national sex and age structure of fiscal year 2022
- `tropical_nights`: population-weighted number of days with daily minimum temperature ≥25 °C in July–September, with land-use correction (main exposure)
- `tropical_nights_uncorrected`: the same without land-use correction
- `tropical_nights_next_summer`, `tropical_nights_prev_summer`: main exposure in the following and previous summers
- `days_min_ge23`, `tmin_mean_jul_aug`: alternative exposure indicators
- `tmax_mean`, `days_max_ge30`, `days_max_ge35`: daytime heat (July–September)
- `vapor_pressure_hpa`: mean daily vapor pressure (July–September)
- `pop_density_per_km2`, `share_65plus`, `share_single_hh`: 2020 Population Census
- `lnight_mean_db`: population-weighted nighttime road traffic noise (https://github.com/m138/japan-night-road-noise)
- `ac_ownership_pct_2009`: prefectural air conditioner ownership (2009 National Survey of Family Income and Expenditure)
- `no2_ppb`: population-weighted traffic-related NO₂ (https://github.com/m138/japan-traffic-no2)

### Columns of `validation_station_year.csv`

`station_type`: 1 = general station, 2 = roadside station, 5 and 9 = other. Estimates include the measurement difference of +0.20 °C (minimum of on-the-hour values). `tropical_nights_land_use_corrected` is the main specification (no intercept). Predictions are from leave-one-prefecture-out cross-validation.

## Code (`src/`)

Python 3.13, pandas, NumPy, SciPy, statsmodels 0.15, geopandas, matplotlib.

1. Outcome: `ndb_q20_multiyear.py` (NDB Open Data tables → area × sex × age counts).
2. Weather: `parse_day.py` (JMA daily minimum temperature), `parse_tmax.py`, `parse_hum.py` (vapor pressure from daily mean temperature and relative humidity).
3. Exposure: `idw.py` (inverse distance squared interpolation with lapse-rate adjustment), `popgrid.py`, `lu_dl.py`, `lu_grid.py`, `lu_lookup.py` (100 m land use), `expo_uhi.py` (land-use corrected tropical nights), `tmax_expo.py`, `hum_expo.py`.
4. Validation: `aq_parse.py` (hourly temperature at air quality monitoring stations), `t10_cmp.py` (minimum of hourly values vs. daily minimum), `uhi_v5.py` (leave-one-prefecture-out validation).
5. Models: `panel_v5.py`, `panel_v5b.py`, `v6.py`, `v6b.py`, `v6c.py` (fixed-effects models, sensitivity analyses, contribution of each year), `em_v7.py` (aggregated binomial model and 2023 scenario).
6. Figures: `en_figs.py` (English Figures 1–4 and Supplementary Figure 1, in `figures/`), `graphical_abstract.py`; `figs.py`, `fig4_v7.py`, and `supp_map.py` are the earlier versions with Japanese labels.

The scripts were run in a working directory with input folders `jma/`, `hn/`, `dem/`, and `lu/`; paths at the top of each script may need to be changed. Population points are from https://github.com/m138/japan-night-road-noise.

## Sources

NDB Open Data (Ministry of Health, Labour and Welfare); Japan Meteorological Agency historical weather data; National Institute for Environmental Studies, Environmental Database (hourly air pollution monitoring data); National Land Numerical Information (land use subdivision mesh L03-b, 2021; elevation G04-a); 2020 Population Census small-area statistics. Access dates are given in the paper.

## License

Code: MIT License. Data in `data/`: CC BY 4.0. Source data remain under the terms of their providers.
