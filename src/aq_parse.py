import pandas as pd,numpy as np,glob,io
L=[]
for f in glob.glob('aq/temp/*_23.txt'):
    t=open(f,'rb').read().decode('cp932','replace')
    d=pd.read_csv(io.StringIO(t),dtype=str); d.columns=[c.strip() for c in d.columns]
    d=d[d['測定項目コード'].str.strip()=='TEMP']
    d['m']=d['測定月'].astype(int); d=d[d.m.isin([7,8,9])]
    H=d[[f'{h:02d}h' for h in range(1,25)]].apply(pd.to_numeric,errors='coerce')
    H=H.where((H>-500)&(H<600))/10
    d['tmin']=H.min(1).where(H.notna().sum(1)>=20)
    L.append(d[['測定年度','測定局コード','m','測定日','tmin']])
D=pd.concat(L); D.columns=['fy','code','m','d','tmin']; D['fy']=D.fy.astype(int)
g=D.groupby(['code','fy'])
S=pd.DataFrame({'n':g.tmin.count(),'tn25':g.tmin.apply(lambda x:(x>=25).sum()),'tmin_ja':D[D.m.isin([7,8])].groupby(['code','fy']).tmin.mean()}).reset_index()
S=S[S.n>=83]   # >=90% of 92 days
S['tn25']=S.tn25*92/S.n
S.to_csv('hn/aq_station_year.csv',index=False); print(S.shape, S.groupby('fy').size().to_dict())
