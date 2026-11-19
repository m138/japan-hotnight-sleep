import pandas as pd,numpy as np,statsmodels.api as sm,sys,itertools,json
sys.path.insert(0,'hn'); from idw import idw_daily
X=pd.read_csv('hn/uhi_station_data.csv',dtype={'code':str,'pref':str,'kind':str})
DELTA=float(sys.argv[1]) if len(sys.argv)>1 else 0.0   # 毎時値の最小と日最低気温の差（°C）
Ts={}
for y in sorted(X.fy.unique()):
    x=X[X.fy==y]; T,ja,W,I,meta=idw_daily(x.lat.values,x.lon.values,x.elev.values,y)
    for i,ix in enumerate(x.index): Ts[ix]=T[i]
X['T']=pd.Series(Ts)
def specs():
    out=[]
    for b1 in ['d_built_500m','d_built_1500m','d_built_4500m']:
        for b2 in [None,'d_built_4500m','d_built_10km']:
            if b2==b1: continue
            for w in [None,'d_water_1500m','d_water_4500m','d_water_10km']:
                out.append([c for c in [b1,b2,w] if c])
    return out
SP=specs()
def lopo_pred(D,cols):
    p=pd.Series(np.nan,index=D.index)
    for pr in D.pref.unique():
        tr=D[D.pref!=pr]; m=sm.OLS(tr.res,sm.add_constant(tr[cols])).fit()
        p[D.pref==pr]=m.predict(sm.add_constant(D.loc[D.pref==pr,cols],has_constant='add'))
    return p
FINAL=['d_built_1500m','d_built_10km','d_water_4500m']
# (a) 最終モデルのLOPO
X['off_final']=lopo_pred(X,FINAL)
# (b) 入れ子LOPO：外側の学習データの中で県LOPOにより仕様を選ぶ
X['off_nested']=np.nan; chosen={}
import os
for pr in ([] if os.environ.get('NONEST') else X.pref.unique()):
    tr=X[X.pref!=pr]
    best=min(SP,key=lambda c: np.sqrt(((tr.res-lopo_pred(tr,c))**2).mean()))
    chosen[pr]=best; m=sm.OLS(tr.res,sm.add_constant(tr[best])).fit()
    X.loc[X.pref==pr,'off_nested']=m.predict(sm.add_constant(X.loc[X.pref==pr,best],has_constant='add'))
if chosen: print('入れ子で選ばれた仕様:',pd.Series({k:'+'.join(v) for k,v in chosen.items()}).value_counts().to_dict())
def tn(off): return np.array([int((t+o+DELTA>=25).sum()) for t,o in zip(X['T'],off)])
X['p_idw']=tn(np.zeros(len(X))); X['p_final']=tn(X.off_final); X['p_nested']=tn(X.off_nested.fillna(0))
# 定数なし（主解析で使った仕様）：各分割で学習した定数を引く
X['off_noc']=np.nan
for pr in X.pref.unique():
    tr=X[X.pref!=pr]; mm=sm.OLS(tr.res,sm.add_constant(tr[FINAL])).fit()
    X.loc[X.pref==pr,'off_noc']=(X.loc[X.pref==pr,FINAL]*mm.params[FINAL]).sum(1)
X['p_final_noc']=tn(X.off_noc)
def stat(o,p): e=p-o; return dict(bias=round(e.mean(),2),rmse=round(np.sqrt((e**2).mean()),2),r2=round(1-((e**2).sum()/((o-o.mean())**2).sum()),3))
def within(o,p,g2):
    d=pd.DataFrame({'o':o,'p':p,'s':X.code,'g':g2})
    for c in ['o','p']:
        d[c+'r']=d[c]
        for it in range(100):
            for g in ['s','g']: d[c+'r']-=d.groupby(g)[c+'r'].transform('mean')
    r=np.corrcoef(d['or'],d['pr'])[0,1]; b=np.polyfit(d['pr'],d['or'],1)[0]
    return dict(r=round(r,3),slope_obs_on_pred=round(b,2),sd_obs=round(d['or'].std(),2),sd_pred=round(d['pr'].std(),2))
R=[]
for lab,sub in [('全局',X.index),('一般局',X.index[X.kind=='1']),('自排局',X.index[X.kind=='2'])]:
    for v in ['p_idw','p_final','p_nested','p_final_noc']:
        R.append(dict(stations=lab,model=v,n=len(sub),**stat(X.loc[sub,'tn25'],X.loc[sub,v])))
R=pd.DataFrame(R); print('DELTA',DELTA); print(R.to_string())
W=[]
X['py']=X.pref+'_'+X.fy.astype(str)
for v in ['p_idw','p_final_noc','p_final']:
    W.append(dict(model=v,fe='局+年度',**within(X.tn25,X[v],X.fy.astype(str))))
    W.append(dict(model=v,fe='局+県×年度',**within(X.tn25,X[v],X.py)))
W=pd.DataFrame(W); print(W.to_string())
print('kind counts',X.kind.value_counts().to_dict(),'stations',X.code.nunique(),'prefs',X.pref.nunique())
R.to_csv(f'hn/uhi_v5_level_d{DELTA}.csv',index=False); W.to_csv(f'hn/uhi_v5_within_d{DELTA}.csv',index=False)
X[['code','pref','fy','kind','tn25','p_idw','p_final_noc','p_final']].to_csv(f'hn/uhi_station_pred_d{DELTA}.csv',index=False)
