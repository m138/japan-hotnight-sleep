import pandas as pd,glob,os,warnings,numpy as np,re
warnings.filterwarnings('ignore')
rows=[]
for f in glob.glob('jma/day/*.html'):
    b=os.path.basename(f)[:-5]; kb,y,m=b.split('_'); k,blk=kb[0],kb[1:]
    try: t=pd.read_html(f)[0]
    except Exception: continue
    cols=[' '.join(map(str,c)) for c in t.columns]
    ic=[i for i,c in enumerate(cols) if '気温' in c and '最低' in c]
    if not ic: continue
    ic=ic[0]
    for _,r in t.iterrows():
        try: d=int(str(r.iloc[0]).strip())
        except: continue
        raw=str(r.iloc[ic]); v=re.sub(r'[\)\]\s]','',raw)
        try: v=float(v)
        except: v=np.nan
        if ']' in raw: v=np.nan   # 資料不足値は除外、準正常値 ) は採用
        rows.append((k,blk,int(y),int(m),d,v))
D=pd.DataFrame(rows,columns=['kind','block','year','month','day','tmin'])
D.to_parquet('hn/tmin_daily.parquet'); print(D.shape, D.tmin.isna().mean())
