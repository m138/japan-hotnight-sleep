import pandas as pd,numpy as np,pyarrow.parquet as pq,sys
sys.path.insert(0,'hn'); from idw import idw_daily; from lu_lookup import feats
b=pd.read_csv('hn/uhi_coef2.csv',index_col=0).iloc[:,0]
cols=['built_1500m','built_10km','water_4500m']
G=pd.read_csv('hn/popgrid_1km.csv',dtype={'code':str}); G['code']=G.code.str.zfill(4)
dem=pd.read_csv('dem/dem_raw.csv',usecols=['G04a_001','G04a_002']); c=dem.G04a_001.astype(str).str.zfill(8)
dem['iy']=c.str[:2].astype(int)*80+c.str[4].astype(int)*10+c.str[6].astype(int)
dem['ix']=(c.str[2:4].astype(int)+100)*80+c.str[5].astype(int)*10+c.str[7].astype(int)
dem['elev']=pd.to_numeric(dem.G04a_002,errors='coerce'); dem.loc[dem.elev<-100,'elev']=np.nan
G=G.merge(dem[['iy','ix','elev']],on=['iy','ix'],how='left'); G['elev']=G.elev.fillna(0); G['cid']=np.arange(len(G))
R=pq.read_table('/mnt/project-files/paper_C_no2/release/data/receptors_no2_noise_2015_2023.parquet',columns=['lat','lon','pop_w']).to_pandas()
R['iy']=np.floor(R.lat*120).astype(int); R['ix']=np.floor(R.lon*80).astype(int)
R=R.merge(G[['iy','ix','cid']],on=['iy','ix'],how='left'); print('receptors',len(R),'unmatched',R.cid.isna().sum(),flush=True)
fr=feats(R.lat.values,R.lon.values)
for k in cols: R[k]=np.nan_to_num(fr[k])
out={}
for y in range(2018,2024):
    T,ja,W,I,meta=idw_daily(G.lat.values,G.lon.values,G.elev.values,y)
    fs=feats(meta.lat.values,meta.lon.values)
    S={k:(W*np.nan_to_num(fs[k])[I]).sum(1) for k in cols}
    cid=R.cid.values.astype(int)
    d=sum(b['d_'+k]*(R[k].values-S[k][cid]) for k in cols)
    for lab,delta in [('u',d),('uc',d+b['const']),('n',np.zeros(len(R)))]:
        q=np.round(delta/0.05).astype(np.int32)
        g=pd.DataFrame({'cid':cid,'q':q,'w':R.pop_w.values}).groupby(['cid','q']).w.sum().reset_index()
        Tc=T[g.cid.values]; dd=(g.q.values*0.05)[:,None]
        tn=((Tc+dd)>=25).sum(1); tj=Tc[:,ja].mean(1)+dd[:,0]
        g['tn']=tn*g.w; g['tj']=tj*g.w; g['code']=G.code.values[g.cid.values]
        a=g.groupby('code')[['w','tn','tj']].sum()
        out[(lab,y)]=pd.DataFrame({f'tn25_{lab}_{y}':a.tn/a.w,f'tmin_ja_{lab}_{y}':a.tj/a.w})
    print(y,'pop-wtd TN: none',round(float((out[('n',y)].iloc[:,0]*a.w).sum()/a.w.sum()),1),' urban',round(float((out[('u',y)].iloc[:,0]*a.w).sum()/a.w.sum()),1),' urban+const',round(float((out[('uc',y)].iloc[:,0]*a.w).sum()/a.w.sum()),1),' mean delta',round(float(np.average(d,weights=R.pop_w)),3),flush=True)
A=pd.concat(out.values(),axis=1); A.index.name='code'; A.to_csv('hn/area_hotnight_uhi.csv')
# national summary: pop share by delta
print('pop-weighted delta quantiles', np.percentile(np.repeat(d,1)[:2000000],[5,25,50,75,95]).round(2))
