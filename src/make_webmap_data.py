"""Make docs/areas.geojson for the web map: tropical nights, insufficient rest from sleep and the 2023 scenario
by secondary medical care area (335 areas, fiscal years 2018-2023).

Usage: python src/make_webmap_data.py AREAS.gpkg data/ docs/areas.geojson
AREAS.gpkg: polygons of the 335 secondary medical care areas with a 4-digit code column 'code'.
"""
import sys, json
import numpy as np, pandas as pd, geopandas as gpd, shapely

bnd, ddir, out = sys.argv[1:4]
YRS = list(range(2018, 2024))
P = pd.read_csv(f'{ddir}/panel_area_year.csv')
N = pd.read_csv(f'{ddir}/ndb_q20_area_sex_age_FY2018-2023.csv', dtype={'area': str}, encoding='utf-8-sig')
nm = N.drop_duplicates('area').set_index('area')[['pref', 'area_nm']]
S = pd.read_csv(f'{ddir}/scenario_2023_by_area.csv')
S['code'] = S.code.astype(str).str.zfill(4)
P['code'] = P.area_code.astype(str).str.zfill(4)
tn = P.pivot(index='code', columns='fiscal_year', values='tropical_nights')
rs = P.pivot(index='code', columns='fiscal_year', values='prop_insufficient_rest_std') * 100
rp = P.pivot(index='code', columns='fiscal_year', values='responses')
G = gpd.read_file(bnd)
G = G.to_crs(4326)
def trim(g, amin=2e-4):  # drop small islands (about 2 km2) unless the area has nothing larger
    parts = sorted(getattr(g, 'geoms', [g]), key=lambda x: -x.area)
    keep = [x for x in parts if x.area >= amin] or parts[:1]
    return shapely.MultiPolygon(keep).simplify(0.0015, preserve_topology=True)
G['geometry'] = G.geometry.apply(trim)
gj = json.loads(G.to_json())
S = S.set_index('code')
feats = []
for f in gj['features']:
    c = f['properties']['code']
    f['properties'] = {'c': c, 'n': f"{nm.loc[c, 'pref']} {nm.loc[c, 'area_nm']}",
                       'resp': int(rp.loc[c, 2023]),
                       'tn': [round(float(tn.loc[c, y]), 1) for y in YRS],
                       'rest': [round(float(rs.loc[c, y]), 1) for y in YRS],
                       'dtn': round(float(S.loc[c, 'dtn']), 1), 'per1000': round(float(S.loc[c, 'per1000']), 2)}
    def rnd(x):
        return [rnd(v) for v in x] if isinstance(x[0], (list, tuple)) else [round(x[0], 3), round(x[1], 3)]
    f['geometry']['coordinates'] = rnd(f['geometry']['coordinates'])
    feats.append(f)
gj['features'] = feats
w = rp
gj['national'] = {'resp': int(rp[2023].sum()),
                  'tn': [round(float(np.average(tn[y], weights=w[y])), 1) for y in YRS],
                  'rest': [round(float(np.average(rs[y], weights=w[y])), 1) for y in YRS],
                  'dtn': round(float(np.average(S.dtn, weights=S.n)), 1),
                  'per1000': round(float(1000 * S.ex.sum() / S.n.sum()), 2)}
json.dump(gj, open(out, 'w'), ensure_ascii=False, separators=(',', ':'))
print(gj['national'])
