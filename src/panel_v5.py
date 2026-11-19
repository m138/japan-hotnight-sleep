exec(open('hn/panel3.py').read().split("def fit(")[0])
import statsmodels.api as sm
U=pd.read_csv('hn/area_hotnight_uhi.csv',dtype={'code':str}); U['code']=U.code.str.zfill(4)
UL=[]
for y in range(2018,2024):
    t=pd.DataFrame({'area':U.code,'year':y,'tn10u':U[f'tn25_u_{y}']/10,'tja_u':U[f'tmin_ja_u_{y}']})
    t['tn10u_lead']=U[f'tn25_u_{y+1}']/10 if y<2023 else np.nan
    t['tn10u_lag']=U[f'tn25_u_{y-1}']/10 if y>2018 else np.nan
    UL.append(t)
D=D.merge(pd.concat(UL),on=['area','year'])
TX=pd.read_csv('hn/area_tmax.csv',dtype={'code':str}); TX['code']=TX.code.str.zfill(4)
D=D.merge(TX.rename(columns={'code':'area'}),on=['area','year'])
H=pd.read_csv('hn/area_hum_no2.csv',dtype={'code':str}); H['code']=H.code.str.zfill(4)
D=D.merge(H[['code','year','e_mean']].rename(columns={'code':'area'}),on=['area','year'])
D['hd30_10']=D.hd30/10; D['hd35_10']=D.hd35/10
D['ry']=D.region+D.year.astype(str); D['py']=D.prefc+D.year.astype(str); D['yr']=D.year.astype(str)
for y in range(2019,2024): D[f'yd{y}']=(D.year==y)*D.ldens
YD=[f'yd{y}' for y in range(2019,2024)]
def fe(data,xs,fes,ctrl=(),lab='',grp='prefc',out=None):
    d=data.dropna(subset=xs).copy(); w=d.tot.values.astype(float)
    cols=['y']+list(xs)+list(ctrl)
    Z=d[cols].astype(float).copy()
    for it in range(300):
        old=Z.copy()
        for g in fes:
            Z-= (Z.mul(w,axis=0)).groupby(d[g].values).transform('sum').div(pd.Series(w,index=d.index).groupby(d[g].values).transform('sum'),axis=0)
        if np.abs(Z-old).max().max()<1e-10: break
    Xm=Z[list(xs)+list(ctrl)]; Xm=Xm.loc[:,Xm.std()>1e-9]
    r=sm.WLS(Z['y'],Xm,weights=w).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d[grp])[0]})
    # 吸収した固定効果の自由度で分散を補正
    N=len(d); k=Xm.shape[1]; G=d[grp].nunique()
    nonnest=sum(d[g].nunique() for g in fes)-len(fes)+1   # 吸収したすべての固定効果を自由度に数える（ダミー変数WLSと同じ、保守的）
    adj=(N-k)/(N-k-nonnest)
    res=[]
    for x in xs:
        b=r.params[x]; se=r.bse[x]*np.sqrt(adj); z=b/se
        from scipy.stats import norm
        p=2*(1-norm.cdf(abs(z)))
        res.append(dict(model=lab,x=x,n=N,OR=round(np.exp(b),4),lo=round(np.exp(b-1.96*se),4),hi=round(np.exp(b+1.96*se),4),p=round(p,4)))
    if out is not None: out.extend(res)
    print(lab,' | '.join(f"{q['x']} {q['OR']} ({q['lo']}-{q['hi']}) p={q['p']}" for q in res),flush=True)
    return r
OUT=[]
# 0) FWLとダミー変数WLSの一致確認
import statsmodels.formula.api as smf
m=smf.wls('y~tn10u+C(area)+C(year)+C(year):ldens',D,weights=D.tot).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(D.prefc)[0]})
print('ダミーWLS',round(np.exp(m.params['tn10u']),5),round(m.bse['tn10u'],6))
r=fe(D,['tn10u'],['area','yr'],YD,'FWL同等確認'); print('FWL',round(np.exp(r.params['tn10u']),5),round(r.bse['tn10u'],6))
MODELS=[('M1 圏域+年度',['area','yr'],[]),('M2 圏域+年度+年度×人口密度（主）',['area','yr'],YD),
        ('M3 圏域+地方×年度+年度×人口密度',['area','ry'],YD),('M4 圏域+県×年度+年度×人口密度',['area','py'],YD),
        ('M3b 圏域+地方×年度（人口密度項なし）',['area','ry'],[]),('M4b 圏域+県×年度（人口密度項なし）',['area','py'],[])]
print('=== 表：モデル比較（全年度／2018除外）')
for lab,fes,ct in MODELS:
    fe(D,['tn10u'],fes,ct,lab+' 全年度',out=OUT); fe(D[D.year!=2018],['tn10u'],fes,ct,lab+' 2018除外',out=OUT)
print('=== 偽の暴露：同じ2018–2022年度で当年と翌年を比較')
d=D[D.year<=2022]
for lab,fes,ct in MODELS[1:4]:
    fe(d,['tn10u'],fes,ct,lab+' 当年 2018-22',out=OUT); fe(d,['tn10u_lead'],fes,ct,lab+' 翌年 2018-22',out=OUT); fe(d,['tn10u','tn10u_lead'],fes,ct,lab+' 同時 2018-22',out=OUT)
print('=== 前年の夏（年度前半の回答が参照しうる夏）2019–2023年度')
d=D[D.year>=2019]
for lab,fes,ct in MODELS[1:4]:
    fe(d,['tn10u'],fes,ct,lab+' 当年 2019-23',out=OUT); fe(d,['tn10u_lag'],fes,ct,lab+' 前年 2019-23',out=OUT); fe(d,['tn10u','tn10u_lag'],fes,ct,lab+' 当年+前年 2019-23',out=OUT)
print('=== 日中の暑さ・湿度')
for lab,fes,ct in MODELS[1:4]:
    for xs in [['tmax_mean'],['hd30_10'],['tn10u','tmax_mean'],['tn10u','hd30_10'],['tn10u','hd35_10'],['tn10u','tmax_mean','e_mean']]:
        fe(D,xs,fes,ct,lab+' '+'+'.join(xs),out=OUT)
print('=== 閾値の異なる指標（23°C・25°C、補正なし）を1 SDあたりで')
for v in ['tn25','tn23']:
    D[v+'_sd']=D[v]/ (D[v]-D.groupby('area')[v].transform('mean')).std()
for lab,fes,ct in MODELS[1:4]:
    fe(D,['tn25_sd'],fes,ct,lab+' 25°C(1SD)',out=OUT); fe(D,['tn23_sd'],fes,ct,lab+' 23°C(1SD)',out=OUT)
print('within-area SD (nights): tn25', round((D.tn25-D.groupby('area').tn25.transform('mean')).std(),2),' tn23',round((D.tn23-D.groupby('area').tn23.transform('mean')).std(),2),' tn10u(x10)',round(10*(D.tn10u-D.groupby('area').tn10u.transform('mean')).std(),2))
pd.DataFrame(OUT).to_csv('hn/panel_v5_results.csv',index=False,encoding='utf-8-sig')
D.to_pickle('hn/panel_v5_D.pkl')
