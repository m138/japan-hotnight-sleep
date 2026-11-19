import pandas as pd,numpy as np,statsmodels.api as sm,warnings; warnings.filterwarnings('ignore')
D=pd.read_pickle('hn/panel_v5_D.pkl')
R=pd.read_csv('/mnt/project-files/multiyear_ndb/ndb_q20_FY2018-2023_area_sex_age.csv',dtype={'area':str},encoding='utf-8-sig').dropna(subset=['no','yes']).rename(columns={'fy':'year'})
R=R[R.area!='9999']; R['sa']=R.sex+R.age; R['n']=R.no+R.yes
A=R.merge(D[['area','year','tn10u','ldens','prefc']],on=['area','year'])
print('rows',len(A),'respondents',int(A.n.sum()))
# 1) 集計二項モデル（性・年齢階級別の回答数）：圏域FE＋性年齢×年度FE＋年度×人口密度
X=pd.get_dummies(A[['area']].astype(str),drop_first=True,dtype=float)
Y=pd.get_dummies((A.sa+'_'+A.year.astype(str)),drop_first=False,dtype=float)
Z=pd.DataFrame({f'yd{y}':(A.year==y)*A.ldens for y in range(2019,2024)})
M=pd.concat([A[['tn10u']],X,Y,Z],axis=1).astype(float)
g=sm.GLM(A.no/A.n,M,family=sm.families.Binomial(),var_weights=A.n).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(A.prefc)[0]})
b,se=g.params['tn10u'],g.bse['tn10u']
# 過分散：Pearson χ²/df で尺度を推定してSEを補正
sc=g.pearson_chi2/g.df_resid
print(f'集計二項（クラスターSE）OR {np.exp(b):.4f} ({np.exp(b-1.96*se):.4f}-{np.exp(b+1.96*se):.4f})  Pearson分散比 {sc:.1f}')
# 2) 2023年度の人数相当値：性・年齢階級別の実回答数と各層の観測割合を用いる
b0=0.010449*0+np.log(1.0105)   # 主モデル M2 の係数（10夜あたり）
lo,hi=np.log(1.0029),np.log(1.0182)
base=D[D.year<2023].groupby('area').tn10u.mean()
a23=A[A.year==2023].copy(); a23['dtn']=a23.tn10u-a23.area.map(base)
def excess(bb):
    p=a23.no/a23.n; lp=np.log(p/(1-p)); p0=1/(1+np.exp(-(lp-bb*a23.dtn)))
    return float(((p-p0)*a23.n).sum()), float(np.average(p-p0,weights=a23.n)*100)
for lab,bb in [('点推定',b0),('下限',lo),('上限',hi)]:
    e,pp=excess(bb); print(f'{lab}: 性年齢別 {e:,.0f}人  平均差 {pp:.3f}ポイント')
print('回答者加重の熱帯夜の差（2023−2018–2022年平均、夜）',round(10*np.average(a23.dtn,weights=a23.n),1),'  2023年度回答者',int(a23.n.sum()))
# 旧方式（標準化割合×圏域の回答者数）との比較
d23=D[D.year==2023].copy(); d23['dtn']=d23.tn10u-d23.area.map(base); p=d23.p
old=float(((p-1/(1+np.exp(-(np.log(p/(1-p))-b0*d23.dtn))))*d23.tot).sum()); print('旧方式（標準化割合×回答者数）',f'{old:,.0f}')
