import pandas as pd,numpy as np,statsmodels.api as sm,warnings,geopandas as gpd; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.rcParams['font.family']='IPAGothic'
D=pd.read_pickle('hn/panel_v5_D.pkl')
R0=pd.read_csv('/mnt/project-files/multiyear_ndb/ndb_q20_FY2018-2023_area_sex_age.csv',dtype={'area':str},encoding='utf-8-sig').dropna(subset=['no','yes']).rename(columns={'fy':'year'})
R0=R0[R0.area!='9999']; R0['sa']=R0.sex+R0.age; R0['n']=R0.no+R0.yes
A=R0.merge(D[['area','year','tn10u','ldens','prefc']],on=['area','year']).reset_index(drop=True)
X=pd.get_dummies(A[['area']].astype(str),drop_first=True,dtype=float)
Y=pd.get_dummies((A.sa+'_'+A.year.astype(str)),drop_first=False,dtype=float)
Zd=pd.DataFrame({f'yd{y}':(A.year==y)*A.ldens for y in range(2019,2024)})
M=pd.concat([A[['tn10u']],X,Y,Zd],axis=1).astype(float)
g=sm.GLM(A.no/A.n,M,family=sm.families.Binomial(),var_weights=A.n).fit()
base=D[D.year<2023].groupby('area').tn10u.mean()
i23=A.index[A.year==2023]; M1=M.loc[i23].copy(); M0=M1.copy(); M0['tn10u']=A.loc[i23,'area'].map(base).values
p1=1/(1+np.exp(-(M1.values@g.params.values))); p0=1/(1+np.exp(-(M0.values@g.params.values)))
E=pd.DataFrame({'code':A.loc[i23,'area'].values,'n':A.loc[i23,'n'].values,'ex':A.loc[i23,'n'].values*(p1-p0)}).groupby('code').sum()
E['dtn']=(D[D.year==2023].set_index('area').tn10u-base)*10
E['per1000']=1000*E.ex/E.n; print('total',round(E.ex.sum()),'OR',round(np.exp(g.params['tn10u']),4)); print(E.describe().round(2))
E.round(3).to_csv('/mnt/project-files/paper_E_hotnight/figures/suppfig_excess_by_area_2023.csv',encoding='utf-8-sig')
G=gpd.read_file('/mnt/project-files/paper_B_analysis_2026-11-09/figures/areas_335_simplified.gpkg').merge(E.reset_index(),on='code')
ok=G.code.str[:2]=='47'; G.loc[ok,'geometry']=G.loc[ok,'geometry'].translate(xoff=5.0,yoff=13.0)
fig,axs=plt.subplots(1,2,figsize=(11,5.6))
for ax,col,cm,vmin,vmax,lab,t in [(axs[0],'dtn','RdBu_r',-30,30,'2023年度と2018–2022年度平均の差（日）','a　熱帯夜日数の差'),
                                   (axs[1],'per1000','RdBu_r',-8,8,'健診回答者1,000人あたり（人）','b　休養不足のモデル上の差')]:
    G.plot(column=col,ax=ax,cmap=cm,vmin=vmin,vmax=vmax,edgecolor='white',linewidth=0.15,legend=True,
           legend_kwds={'orientation':'horizontal','shrink':0.7,'pad':0.02,'label':lab,'extend':'both'})
    ax.plot([128.5,136.6,136.6],[36.6,36.6,40.6],color='#bbbbbb',lw=0.8); ax.text(128.7,40.2,'沖縄',fontsize=8)
    ax.set_xlim(128.3,146.2); ax.set_ylim(30.0,45.8); ax.set_axis_off(); ax.set_title(t,loc='left',fontsize=11)
fig.savefig('/mnt/project-files/paper_E_hotnight/figures/suppfig_excess_map.png',dpi=300,bbox_inches='tight')
