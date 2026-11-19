import pandas as pd,numpy as np,glob,os,re,warnings
from multiprocessing import Pool
warnings.filterwarnings('ignore')
def num(x):
    x=str(x)
    if ']' in x: return np.nan
    try: return float(re.sub(r'[\)\]\s]','',x))
    except: return np.nan
def one(f):
    b,y,m,d=os.path.basename(f)[:-5].split('_')
    try: t=pd.read_html(f)[0]
    except Exception: return None
    T=np.array([num(v) for v in t.iloc[:,4]])
    if len(T)!=144 or np.isnan(T).sum()>6: return None
    H=T.reshape(24,6)
    return dict(block=b,year=int(y),month=int(m),day=int(d),min10=np.nanmin(T),min_hinst=np.nanmin(H[:,5]),min_hmean=np.nanmin(np.nanmean(H,1)))
if __name__=='__main__':
    fs=[f for f in glob.glob('jma/t10/*.html') if os.path.getsize(f)>5000]
    with Pool(4) as p: r=[x for x in p.map(one,fs,chunksize=50) if x]
    C=pd.DataFrame(r)
    T=pd.read_parquet('hn/tmin_daily.parquet'); T=T[T.kind=='s']
    C=C.merge(T[['block','year','month','day','tmin']],on=['block','year','month','day']).dropna()
    g=C.groupby('block'); full=g.size(); C=C[C.block.isin(full[full>=83].index)]
    print('stations',C.block.nunique(),'days',len(C))
    for v in ['min10','min_hinst','min_hmean']:
        dlt=C[v]-C.tmin
        S=C.groupby('block').apply(lambda d: pd.Series({'tn_off':(d.tmin>=25).sum()*92/len(d),'tn_v':(d[v]>=25).sum()*92/len(d)}))
        S['diff']=S.tn_v-S.tn_off
        print(f'{v}: 日最低気温との差 平均 {dlt.mean():.2f} °C (中央値 {dlt.median():.2f}, 90%点 {dlt.quantile(.9):.2f})  熱帯夜日数の差 平均 {S["diff"].mean():.1f}日 (範囲 {S["diff"].min():.1f}〜{S["diff"].max():.1f})  公式TN平均 {S.tn_off.mean():.1f}')
    C.to_csv('hn/t10_cmp.csv',index=False)
