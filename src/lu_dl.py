import pandas as pd,subprocess,os,zipfile
from dbfread import DBF
import numpy as np
g=pd.read_csv('../hn/popgrid_1km.csv')
m1=sorted(set((g.iy//80).astype(int).astype(str)+(g.ix//80-100).astype(int).astype(str)))
for m in m1:
    out=f'lu_{m}.parquet'
    if os.path.exists(out): continue
    r=subprocess.run(['curl','-s','-m','600','-o','tmp.zip','-w','%{http_code}',f'https://nlftp.mlit.go.jp/ksj/gml/data/L03-b/L03-b-21/L03-b-21_{m}-jgd2011_GML.zip'],capture_output=True,text=True).stdout
    if r!='200': print('miss',m,r,flush=True); continue
    z=zipfile.ZipFile('tmp.zip'); n=[x for x in z.namelist() if x.endswith('.dbf')][0]; z.extract(n,'.')
    d=pd.DataFrame(iter(DBF(n))); os.remove(n)
    d.columns=['code','lu','date']; d=d[['code','lu']]; d['code']=d.code.astype('int64'); d['lu']=d.lu.astype(int)
    d.to_parquet(out); print(m,len(d),flush=True)
os.remove('tmp.zip'); print('done')
