import pandas as pd,numpy as np
from scipy.spatial import cKDTree
LAPSE=0.0065
st=pd.read_csv('jma/stations.csv',dtype=str); st[['lat','lon','h']]=st[['lat','lon','h']].astype(float); st=st[(st.lat>20)&(st.lon>120)&(st.lon<155)]
D=pd.read_parquet('hn/tmin_daily.parquet').merge(st[['kind','block','lat','lon','h']],on=['kind','block'])
def xyz(lat,lon):
    la,lo=np.radians(lat),np.radians(lon); return np.c_[np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)]*6371
_cache={}
def year_stations(y):
    if y in _cache: return _cache[y]
    d=D[(D.year==y)&(D.month.isin([7,8,9]))].copy(); d['t0']=d.tmin+LAPSE*d.h
    P=d.pivot_table(index=['kind','block'],columns=['month','day'],values='t0')
    ok=P.notna().mean(1)>=0.9; P=P[ok]
    meta=st.set_index(['kind','block']).loc[P.index]
    ja=np.array([m in (7,8) for m,dd in P.columns])
    _cache[y]=(P.values.astype(np.float32),meta,cKDTree(xyz(meta.lat.values,meta.lon.values)),ja)
    return _cache[y]
def idw_daily(lat,lon,elev,y,k=6,chunk=200000):
    V,meta,tree,ja=year_stations(y)
    out=np.empty((len(lat),V.shape[1]),np.float32); W=np.empty((len(lat),k),np.float32); I=np.empty((len(lat),k),int)
    for s in range(0,len(lat),chunk):
        sl=slice(s,s+chunk)
        dist,idx=tree.query(xyz(np.asarray(lat)[sl],np.asarray(lon)[sl]),k=k)
        w=1/np.maximum(dist,1.0)**2
        num=np.zeros((len(dist),V.shape[1]),np.float32); den=np.zeros_like(num)
        for j in range(k):
            v=V[idx[:,j]]; m=~np.isnan(v); num+=np.where(m,v,0)*w[:,j:j+1]; den+=m*w[:,j:j+1]
        out[sl]=num/den - LAPSE*np.asarray(elev)[sl][:,None]
        W[sl]=w/w.sum(1,keepdims=True); I[sl]=idx
    return out,ja,W,I,meta
