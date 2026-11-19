import pandas as pd,glob,os,warnings,numpy as np,re
from multiprocessing import Pool
warnings.filterwarnings('ignore')
def num(raw):
    raw=str(raw)
    if ']' in raw: return np.nan
    try: return float(re.sub(r'[\)\]\s]','',raw))
    except: return np.nan
def one(f):
    b=os.path.basename(f)[:-5]; kb,y,m=b.split('_'); k,blk=kb[0],kb[1:]
    if int(m) not in (7,8,9): return []
    try: t=pd.read_html(f)[0]
    except Exception: return []
    cols=[' '.join(map(str,c)) for c in t.columns]
    ih=[i for i,c in enumerate(cols) if c.startswith('気温') and '最高' in c]
    it=[i for i,c in enumerate(cols) if c.startswith('気温') and '平均' in c]
    if not ih or not it: return []
    out=[]
    for _,r in t.iterrows():
        try: d=int(str(r.iloc[0]).strip())
        except: continue
        out.append((k,blk,int(y),int(m),d,num(r.iloc[it[0]]),num(r.iloc[ih[0]])))
    return out
if __name__=='__main__':
    fs=glob.glob('jma/day/*.html')
    with Pool(8) as p: res=p.map(one,fs,chunksize=50)
    D=pd.DataFrame([x for r in res for x in r],columns=['kind','block','year','month','day','tmean','tmax'])

    D.to_parquet('hn/tmax_daily.parquet')
    s=D.dropna(subset=['tmax']).groupby('year')[['kind','block']].apply(lambda x: len(x.drop_duplicates()))
    print(D.shape, D.tmax.isna().mean()); print(s)
