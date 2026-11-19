import pandas as pd,numpy as np,pyarrow.parquet as pq,sys
from scipy.spatial import cKDTree
sys.path.insert(0,'hn'); from idw import idw_daily, xyz, st; from lu_lookup import feats
H=pd.read_parquet('hn/hum_daily.parquet').dropna(subset=['e']).merge(st[['kind','block']],on=['kind','block'])
# 6年すべてで湿度がある地点に固定（年ごとの地点数の増加で精度が変わらないように）
k6=H.groupby(['kind','block']).year.nunique(); keep=k6[k6==6].index
H=H.set_index(['kind','block']).loc[keep].reset_index()
b=pd.read_csv('hn/uhi_coef2.csv',index_col=0).iloc[:,0]; cols=['built_1500m','built_10km','water_4500m']
G=pd.read_csv('hn/popgrid_1km.csv',dtype={'code':str}); G['code']=G.code.str.zfill(4)
dem=pd.read_csv('dem/dem_raw.csv',usecols=['G04a_001','G04a_002']); c=dem.G04a_001.astype(str).str.zfill(8)
dem['iy']=c.str[:2].astype(int)*80+c.str[4].astype(int)*10+c.str[6].astype(int)
dem['ix']=(c.str[2:4].astype(int)+100)*80+c.str[5].astype(int)*10+c.str[7].astype(int)
dem['elev']=pd.to_numeric(dem.G04a_002,errors='coerce'); dem.loc[dem.elev<-100,'elev']=np.nan
G=G.merge(dem[['iy','ix','elev']],on=['iy','ix'],how='left'); G['elev']=G.elev.fillna(0); G['cid']=np.arange(len(G))
R=pq.read_table('/mnt/project-files/paper_C_no2/release/data/receptors_no2_noise_2015_2023.parquet',columns=['lat','lon','pop_w']+[f'NO2_{y}' for y in range(2018,2024)]).to_pandas()
R['iy']=np.floor(R.lat*120).astype(int); R['ix']=np.floor(R.lon*80).astype(int)
R=R.merge(G[['iy','ix','cid']],on=['iy','ix'],how='left'); R=R.dropna(subset=['cid'])
fr=feats(R.lat.values,R.lon.values)
for k in cols: R[k]=np.nan_to_num(fr[k])
cid=R.cid.values.astype(int); R['code']=G.code.values[cid]
# 圏域・年の人口加重NO2
no2=R.groupby('code').apply(lambda d: pd.Series({y:np.average(d[f'NO2_{y}'],weights=d.pop_w) for y in range(2018,2024)}))
def es(T): return 6.112*np.exp(17.67*T/(T+243.5))
def DI(T,rh): return 0.81*T+0.01*rh*(0.99*T-14.3)+46.3
out=[]
for y in range(2018,2024):
    T,ja,W,I,meta=idw_daily(G.lat.values,G.lon.values,G.elev.values,y)
    fs=feats(meta.lat.values,meta.lon.values)
    S={k:(W*np.nan_to_num(fs[k])[I]).sum(1) for k in cols}
    d=sum(b['d_'+k]*(R[k].values-S[k][cid]) for k in cols)
    # 日別の水蒸気圧を地点→1 kmへIDW（標高補正なし）
    h=H[H.year==y]; P=h.pivot_table(index=['kind','block'],columns=['month','day'],values='e')
    P=P.reindex(columns=pd.MultiIndex.from_tuples([(m,dd) for m in (7,8,9) for dd in range(1,32) if not (m==9 and dd==31)]))
    m2=st.set_index(['kind','block']).loc[P.index]
    tr=cKDTree(xyz(m2.lat.values,m2.lon.values)); dist,idx=tr.query(xyz(G.lat.values,G.lon.values),k=6)
    w=1/np.maximum(dist,1.0)**2; V=P.values.astype(np.float32)
    num=np.zeros((len(G),V.shape[1]),np.float32); den=np.zeros_like(num)
    for j in range(6):
        v=V[idx[:,j]]; mk=~np.isnan(v); num+=np.where(mk,v,0)*w[:,j:j+1]; den+=mk*w[:,j:j+1]
    E=num/den
    assert E.shape==T.shape, (E.shape,T.shape)
    q=np.round(d/0.05).astype(np.int32)
    g=pd.DataFrame({'cid':cid,'q':q,'w':R.pop_w.values}).groupby(['cid','q']).w.sum().reset_index()
    Tc=T[g.cid.values]+(g.q.values*0.05)[:,None]; Ec=E[g.cid.values]
    rh=np.clip(Ec/es(Tc)*100,0,100)
    di=DI(Tc,rh)
    r=pd.DataFrame({'code':G.code.values[g.cid.values],'w':g.w.values,
        'tn25':(Tc>=25).sum(1),'di75':(di>=75).sum(1),'e_mean':np.nanmean(Ec,1),'rh_tmin':np.nanmean(rh,1),
        'tn25_humid':((Tc>=25)&(rh>=90)).sum(1),'tn25_dry':((Tc>=25)&(rh<90)).sum(1)})
    for c2 in r.columns[2:]: r[c2]=r[c2]*r.w
    a=r.groupby('code').sum(); a=a.div(a.w,axis=0).drop(columns='w'); a['year']=y; a['no2']=no2[y]
    out.append(a.reset_index())
    print(y,{k:round(float(np.average(a[k],weights=None)),2) for k in ['tn25','di75','e_mean','rh_tmin','tn25_humid','no2']},flush=True)
pd.concat(out).to_csv('hn/area_hum_no2.csv',index=False)
