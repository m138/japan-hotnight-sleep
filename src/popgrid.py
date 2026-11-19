import pyarrow.parquet as pq,pandas as pd,numpy as np,geopandas as gpd
t=pq.read_table('/mnt/project-files/paper_C_no2/release/data/receptors_no2_noise_2015_2023.parquet',columns=['lon','lat','pop_w','Lnight_2021','NO2_2022']).to_pandas()
t['iy']=np.floor(t.lat*120).astype(int); t['ix']=np.floor(t.lon*80).astype(int)  # 30" x 45" = 3rd mesh
t['wL']=t.pop_w*t.Lnight_2021; t['wN']=t.pop_w*t.NO2_2022
g=t.groupby(['iy','ix']).agg(pop=('pop_w','sum'),wL=('wL','sum'),wN=('wN','sum')).reset_index()
g['lat']=(g.iy+0.5)/120; g['lon']=(g.ix+0.5)/80
a=gpd.read_file('/mnt/project-files/paper_B_analysis_2026-11-09/figures/areas_335_simplified.gpkg')
p=gpd.GeoDataFrame(g,geometry=gpd.points_from_xy(g.lon,g.lat),crs=4326)
j=gpd.sjoin(p,a,predicate='within',how='left'); j=j[~j.index.duplicated()]
miss=j.code.isna()
if miss.any():
    jn=gpd.sjoin_nearest(p.loc[j.index[miss]].to_crs(6677),a.to_crs(6677),how='left')
    jn=jn[~jn.index.duplicated()]
    j.loc[miss,'code']=jn.code.reindex(j.index[miss]).values
j=j.drop(columns=['geometry','index_right'],errors='ignore')
j.to_csv('hn/popgrid_1km.csv',index=False)
print(len(j), j['pop'].sum(), miss.sum())
