"""Map for the README: tropical nights in July-September 2023 by secondary medical care area.

Usage: python src/make_readme_map.py AREAS.gpkg data/panel_area_year.csv docs/images/map_tropical_nights_2023.png
AREAS.gpkg: polygons of the 335 secondary medical care areas with a 4-digit code column 'code'.
"""
import sys, pandas as pd, geopandas as gpd, shapely
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BND, PANEL, OUT = sys.argv[1:4]
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9})
P = pd.read_csv(PANEL)
P = P[P.fiscal_year == 2023].assign(code=lambda d: d.area_code.astype(str).str.zfill(4))
G = gpd.read_file(BND).to_crs(6677).merge(P[['code', 'tropical_nights']], on='code', how='left')
bins = [0, 1, 10, 20, 30, 45, 1000]
cols = ['#f1efe8', '#fbd9b8', '#f5ab79', '#e6743f', '#c2471d', '#7f2704']
G['cls'] = pd.cut(G.tropical_nights, bins, include_lowest=True, right=False, labels=False)
isl = G.code.str.startswith('47')


def draw(a, D):
    for i, c in enumerate(cols):
        D[D.cls == i].plot(ax=a, color=c, lw=0)
    D.boundary.plot(ax=a, color='white', lw=0.15)
    a.set_axis_off(); a.set_aspect('equal')


fig, ax = plt.subplots(figsize=(6.5, 7))
draw(ax, G[~isl])
xmin, ymin, xmax, ymax = G[~isl & ~G.code.isin(['1313'])].total_bounds   # exclude the Tokyo islands from the extent
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ia = ax.inset_axes([0.04, 0.66, 0.2, 0.26]); draw(ia, G[isl])
bb = gpd.GeoSeries([shapely.box(127.6, 26.05, 128.35, 26.9)], crs=4326).to_crs(6677).total_bounds
ia.set_xlim(bb[0], bb[2]); ia.set_ylim(bb[1], bb[3]); ia.set_axis_on(); ia.set_xticks([]); ia.set_yticks([]); ia.set_xlabel(''); ia.set_ylabel('')
for sp in ia.spines.values():
    sp.set_color('#c3c2b7'); sp.set_visible(True)
ia.set_title('Okinawa Island', fontsize=7, loc='left')
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
ax.legend(h, ['<1', '1–10', '10–20', '20–30', '30–45', '≥45'],
          title='Tropical nights, July–September 2023\n(days with minimum ≥25 °C; population-\nweighted, land-use corrected; by area)',
          loc='lower right', frameon=False, fontsize=7, title_fontsize=8)
fig.tight_layout(); fig.savefig(OUT, dpi=300); plt.close()
