import pandas as pd,numpy as np,pyarrow.parquet as pq,sys
sys.path.insert(0,'hn'); import idw
T=pd.read_parquet('hn/tmax_daily.parquet').rename(columns={'tmax':'tmin'})[['kind','block','year','month','day','tmin']]
idw.D=T.merge(idw.st[['kind','block','lat','lon','h']],on=['kind','block']); idw._cache.clear()
G=pd.read_csv('hn/popgrid_1km.csv',dtype={'code':str}); G['code']=G.code.str.zfill(4)
dem=pd.read_csv('dem/dem_raw.csv',usecols=['G04a_001','G04a_002']); c=dem.G04a_001.astype(str).str.zfill(8)
dem['iy']=c.str[:2].astype(int)*80+c.str[4].astype(int)*10+c.str[6].astype(int)
dem['ix']=(c.str[2:4].astype(int)+100)*80+c.str[5].astype(int)*10+c.str[7].astype(int)
dem['elev']=pd.to_numeric(dem.G04a_002,errors='coerce'); dem.loc[dem.elev<-100,'elev']=np.nan
G=G.merge(dem[['iy','ix','elev']],on=['iy','ix'],how='left'); G['elev']=G.elev.fillna(0); G['cid']=np.arange(len(G))
R=pq.read_table('/mnt/project-files/paper_C_no2/release/data/receptors_no2_noise_2015_2023.parquet',columns=['lat','lon','pop_w']).to_pandas()
R['iy']=np.floor(R.lat*120).astype(int); R['ix']=np.floor(R.lon*80).astype(int)
R=R.merge(G[['iy','ix','cid']],on=['iy','ix']); w=R.groupby('cid').pop_w.sum()
G['w']=G.cid.map(w).fillna(0); G=G[G.w>0]
out=[]
for y in range(2018,2024):
    Tm,ja,W,I,meta=idw.idw_daily(G.lat.values,G.lon.values,G.elev.values,y)
    r=pd.DataFrame({'code':G.code.values,'w':G.w.values,'tmax_mean':np.nanmean(Tm,1),'hd30':(Tm>=30).sum(1),'hd35':(Tm>=35).sum(1)})
    for c2 in ['tmax_mean','hd30','hd35']: r[c2]*=r.w
    a=r.groupby('code').sum(); a=a.div(a.w,axis=0).drop(columns='w'); a['year']=y; out.append(a.reset_index())
    print(y,a[['tmax_mean','hd30','hd35']].mean().round(2).to_dict(),flush=True)
pd.concat(out).to_csv('hn/area_tmax.csv',index=False)
