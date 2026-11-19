import pandas as pd,numpy as np,statsmodels.api as sm,warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.rcParams['font.family']='IPAGothic'
D=pd.read_pickle('hn/panel_v5_D.pkl')
YD=[f'yd{y}' for y in range(2019,2024)]
def resid(d,cols,fes):
    w=d.tot.values.astype(float); Z=d[cols].astype(float).copy()
    for it in range(300):
        old=Z.copy()
        for g in fes:
            Z-=(Z.mul(w,axis=0)).groupby(d[g].values).transform('sum').div(pd.Series(w,index=d.index).groupby(d[g].values).transform('sum'),axis=0)
        if np.abs(Z-old).max().max()<1e-11: break
    return Z
def partial(d,fes,x='tn10u'):
    Z=resid(d,['y',x]+YD,fes); w=d.tot.values
    # 年度×人口密度も部分化（FWL）
    out={}
    for c in ['y',x]:
        b=sm.WLS(Z[c],Z[YD],weights=w).fit(); out[c]=Z[c]-b.fittedvalues
    return pd.DataFrame(out)
# --- 図：M2・M4の調整後散布図
fig,axs=plt.subplots(1,2,figsize=(10,4.4),sharey=False)
C_OTHER,C_2018='#2a78d6','#eb6834'
for ax,(lab,fes) in zip(axs,[('M2 圏域＋年度＋年度×人口密度',['area','yr']),('M4 圏域＋県×年度＋年度×人口密度',['area','py'])]):
    P=partial(D,fes); P['year']=D.year.values; P['w']=D.tot.values
    xs=P.tn10u*10; ys=P.y*100
    s=20+80*P.w/P.w.max()
    for yr,c,nm,z in [(False,C_OTHER,'2019–2023年度',2),(True,C_2018,'2018年度',3)]:
        m=(P.year==2018)==yr
        ax.scatter(xs[m],ys[m],s=s[m],c=c,alpha=0.55,edgecolors='white',linewidths=0.6,label=nm,zorder=z)
    xx=np.linspace(xs.min(),xs.max(),10)
    for sub,ls,nm in [(P,'-','全年度'),(P[P.year!=2018],'--','2018年度の点を除いた')]:
        b=sm.WLS(sub.y*100,sub.tn10u*10,weights=sub.w).fit().params.iloc[0]
        ax.plot(xx,b*xx,ls,color='#333333',lw=2,label=f'{nm}の回帰線（OR {np.exp(b/100*10):.3f}/10夜）',zorder=4)
    ax.axhline(0,color='#bbbbbb',lw=0.8,zorder=1); ax.axvline(0,color='#bbbbbb',lw=0.8,zorder=1)
    ax.set_title(lab,fontsize=10.5); ax.set_xlabel('調整後の熱帯夜日数（日）'); ax.set_ylabel('調整後の休養不足（ロジット×100）')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8,frameon=False,loc='upper left')
    # 2018年度の説明力
    sd=P.groupby(P.year==2018).apply(lambda g: pd.Series({'sd_x':np.sqrt(np.average(g.tn10u**2*100,weights=g.w)),'n':len(g)}))
    print(lab,'\n',sd.round(2))
    top=P.assign(area=D.area.values,area_nm=D.get('area_nm',D.area).values).loc[P.year==2018].assign(ax=lambda d:d.tn10u*10).sort_values('ax')
    print(' 2018年度で調整後の熱帯夜が大きい圏域 上位5:',top.tail(5)[['area','ax','y']].round(3).to_dict('records'))
    print(' 下位5:',top.head(5)[['area','ax','y']].round(3).to_dict('records'))
plt.tight_layout(); plt.savefig('hn/fig_scatter_2018.png',dpi=200)
# 2018年度の調整後熱帯夜が大きい圏域の県
P=partial(D,['area','yr']); P['year']=D.year.values; P['pref']=D.prefc.values
g=P[P.year==2018].assign(x=lambda d:d.tn10u*10).groupby('pref').x.mean().sort_values()
print('M2で2018年度の調整後熱帯夜（県平均）上位',g.tail(8).round(1).to_dict(),' 下位',g.head(5).round(1).to_dict())
# --- 昼夜指標の固定効果除去後の相関
rows=[]
for lab,fes in [('圏域＋年度',['area','yr']),('圏域＋県×年度',['area','py'])]:
    Z=resid(D,['tn10u','tmax_mean','hd30_10','hd35_10','e_mean']+YD,fes)
    R={}
    for c in ['tn10u','tmax_mean','hd30_10','hd35_10','e_mean']:
        R[c]=Z[c]-sm.WLS(Z[c],Z[YD],weights=D.tot).fit().fittedvalues
    R=pd.DataFrame(R); w=D.tot.values
    def wc(a,b): 
        a=a-np.average(a,weights=w); b=b-np.average(b,weights=w); return np.average(a*b,weights=w)/np.sqrt(np.average(a*a,weights=w)*np.average(b*b,weights=w))
    rows.append(dict(FE=lab,**{f'熱帯夜×{c}':round(wc(R.tn10u,R[c]),2) for c in ['tmax_mean','hd30_10','hd35_10','e_mean']}))
print(pd.DataFrame(rows).to_string())
# --- 集計二項モデル内での人数相当値
R0=pd.read_csv('/mnt/project-files/multiyear_ndb/ndb_q20_FY2018-2023_area_sex_age.csv',dtype={'area':str},encoding='utf-8-sig').dropna(subset=['no','yes']).rename(columns={'fy':'year'})
R0=R0[R0.area!='9999']; R0['sa']=R0.sex+R0.age; R0['n']=R0.no+R0.yes
A=R0.merge(D[['area','year','tn10u','ldens','prefc']],on=['area','year']).reset_index(drop=True)
X=pd.get_dummies(A[['area']].astype(str),drop_first=True,dtype=float)
Y=pd.get_dummies((A.sa+'_'+A.year.astype(str)),drop_first=False,dtype=float)
Zd=pd.DataFrame({f'yd{y}':(A.year==y)*A.ldens for y in range(2019,2024)})
M=pd.concat([A[['tn10u']],X,Y,Zd],axis=1).astype(float)
g=sm.GLM(A.no/A.n,M,family=sm.families.Binomial(),var_weights=A.n).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(A.prefc)[0]})
b,se=g.params['tn10u'],g.bse['tn10u']; print(f'集計二項 OR {np.exp(b):.4f} ({np.exp(b-1.96*se):.4f}-{np.exp(b+1.96*se):.4f})')
base=D[D.year<2023].groupby('area').tn10u.mean()
i23=A.index[A.year==2023]; M1=M.loc[i23].copy(); M0=M1.copy(); M0['tn10u']=A.loc[i23,'area'].map(base).values
eta1=M1.values@g.params.values; eta0=M0.values@g.params.values
def ex(bb):
    e1=eta1+(bb-b)*M1.tn10u.values; e0=eta0+(bb-b)*M0.tn10u.values
    p1=1/(1+np.exp(-e1)); p0=1/(1+np.exp(-e0)); n=A.loc[i23,'n'].values
    return (n*(p1-p0)).sum(), 100*np.average(p1-p0,weights=n)
for nm,bb in [('点推定',b),('下限',b-1.96*se),('上限',b+1.96*se)]:
    e,pp=ex(bb); print(f'{nm}: {e:,.0f}人  {pp:.3f}ポイント')
