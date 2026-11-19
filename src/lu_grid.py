# 100 m land use (L03-b-21) -> 500 m class counts -> moving-window shares; point lookup helper
import pandas as pd,numpy as np,glob
from scipy.ndimage import uniform_filter
Y0,X0=24*240,122*160; NY,NX=(46-24)*240,(154-122)*160
cls={'built':[700,901,902,1000],'green':[100,200,500,600,1600],'water':[1100,1400,1500],'sea':[1500]}
A={k:np.zeros((NY,NX),np.float32) for k in list(cls)+['all']}
for f in glob.glob('lu/lu_*.parquet'):
    d=pd.read_parquet(f); c=d.code.astype(str).str.zfill(10)
    iy=c.str[:2].astype(int)*800+c.str[4].astype(int)*100+c.str[6].astype(int)*10+c.str[8].astype(int)
    ix=(c.str[2:4].astype(int)+100)*800+c.str[5].astype(int)*100+c.str[7].astype(int)*10+c.str[9].astype(int)
    gy=(iy//5-Y0).values; gx=(ix//5-X0).values
    ok=(gy>=0)&(gy<NY)&(gx>=0)&(gx<NX)
    np.add.at(A['all'],(gy[ok],gx[ok]),1)
    for k,v in cls.items():
        m=ok&d.lu.isin(v).values; np.add.at(A[k],(gy[m],gx[m]),1)
F={}
for size,lab in [(1,'500m'),(3,'1500m'),(9,'4500m'),(21,'10km')]:
    tot=uniform_filter(A['all'],size,mode='constant')
    for k in cls:
        if lab=='500m' and k=='sea': continue
        F[f'{k}_{lab}']=np.where(tot>0,uniform_filter(A[k],size,mode='constant')/np.maximum(tot,1e-9),np.nan).astype(np.float32)
np.savez_compressed('hn/lu_features.npz',**F)
print({k:float(np.nanmean(v)) for k,v in F.items()})
