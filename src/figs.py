import pandas as pd,numpy as np,geopandas as gpd,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import rcParams; rcParams['font.family']='IPAGothic'; rcParams['axes.spines.top']=False; rcParams['axes.spines.right']=False
OUT='/mnt/project-files/paper_E_hotnight/figures/'
INK='#333333'; C1='#2a78d6'; C2='#eb6834'; GRID='#bbbbbb'
D=pd.read_pickle('panel_v5_D.pkl'); D['tn25']=D.tn10u*10
# ---------- 図1 ----------
A=pd.read_csv('area_hotnight_uhi.csv',dtype={'code':str})
g=gpd.read_file('/mnt/project-files/paper_B_analysis_2026-11-09/figures/areas_335_simplified.gpkg').merge(A,on='code')
ok=g.code.str[:2]=='47'
g.loc[ok,'geometry']=g.loc[ok,'geometry'].translate(xoff=5.0,yoff=13.0)   # 沖縄を枠内へ移動
fig=plt.figure(figsize=(12,5.2)); gs=fig.add_gridspec(1,3,width_ratios=[1,1,0.85],wspace=0.08)
vmax=60; cmap=plt.get_cmap('OrRd')
for i,yr in enumerate([2021,2023]):
    ax=fig.add_subplot(gs[0,i]); g.plot(column=f'tn25_u_{yr}',ax=ax,cmap=cmap,vmin=0,vmax=vmax,edgecolor='white',linewidth=0.15)
    ax.plot([128.5,136.6,136.6],[36.6,36.6,40.6],color=GRID,lw=0.8); ax.text(128.7,40.2,'沖縄',fontsize=8,color=INK)
    ax.set_xlim(128.3,146.2); ax.set_ylim(30.0,45.8); ax.set_axis_off(); ax.set_title(f'{"ab"[i]}　{yr}年 7–9月',loc='left',fontsize=11)
sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,vmax)); cax=fig.add_axes([0.14,0.08,0.4,0.025])
cb=fig.colorbar(sm,cax=cax,orientation='horizontal',extend='max'); cb.set_label('熱帯夜日数（圏域の人口加重、都市補正）',fontsize=9)
ax=fig.add_subplot(gs[0,2]); ax.set_position([0.70,0.2,0.27,0.62])
m=D.groupby('year').apply(lambda d:np.average(d.tn25,weights=d.tot)); print(m.round(1).to_dict())
ax.bar(m.index,m.values,color=C2,width=0.6); 
for x,v in m.items(): ax.text(x,v+0.6,f'{v:.1f}',ha='center',fontsize=8.5,color=INK)
ax.set_xticks(m.index,[f'{y}' for y in m.index],fontsize=8.5); ax.set_ylabel('熱帯夜日数（全国平均）'); ax.set_ylim(0,38)
ax.set_title('c　全国平均の年次推移',loc='left',fontsize=11)
ax.text(0,-0.2,'圏域別の人口加重値を健診回答数で平均',transform=ax.transAxes,fontsize=8,color='#666666')
fig.savefig(OUT+'fig1_hotnight_map.png',dpi=300,bbox_inches='tight'); plt.close()
# ---------- 図2 ----------
P=pd.read_csv('uhi_station_pred_d0.2.csv'); P['kind']=P.kind.astype(str)
fig,axs=plt.subplots(1,3,figsize=(13,4.4))
def lvl(ax,col,t):
    for k,c,lab in [('1',C1,'一般局'),('2',C2,'自排局')]:
        s=P[P.kind==k]; ax.scatter(s[col],s.tn25,s=8,c=c,alpha=0.45,edgecolors='none',label=lab)
    s=P[~P.kind.isin(['1','2'])]; ax.scatter(s[col],s.tn25,s=8,c='#999999',alpha=0.45,edgecolors='none',label='その他')
    e=P[col]-P.tn25; r2=1-(e**2).sum()/((P.tn25-P.tn25.mean())**2).sum()
    ax.plot([0,100],[0,100],color=INK,lw=1); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect('equal')
    ax.set_xlabel('推計の熱帯夜日数（日）'); ax.set_ylabel('常時監視局で測った熱帯夜日数（日）')
    ax.text(0.04,0.96,f'R² {r2:.2f}\n偏り {e.mean():+.1f}日\nRMSE {np.sqrt((e**2).mean()):.1f}日',transform=ax.transAxes,va='top',fontsize=9,color=INK)
    ax.set_title(t,loc='left',fontsize=10.5)
lvl(axs[0],'p_idw','a　気象観測所の内挿のみ'); lvl(axs[1],'p_final_noc','b　土地利用補正（主解析の仕様）')
axs[0].legend(loc='lower right',frameon=False,fontsize=8.5,markerscale=2)
# c 年変動
d=P.copy(); d['py']=d.pref.astype(str)+'_'+d.fy.astype(str)
for c in ['tn25','p_idw','p_final_noc']:
    d[c+'r']=d[c].astype(float)
    for it in range(100):
        for gcol in ['code','py']: d[c+'r']-=d.groupby(gcol)[c+'r'].transform('mean')
ax=axs[2]; ax.scatter(d.p_final_nocr,d.tn25r,s=8,c=C1,alpha=0.45,edgecolors='none')
r0=np.corrcoef(d.p_idwr,d.tn25r)[0,1]; r1=np.corrcoef(d.p_final_nocr,d.tn25r)[0,1]; b=np.polyfit(d.p_final_nocr,d.tn25r,1)[0]
lim=25; ax.plot([-lim,lim],[-lim,lim],color=INK,lw=1); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
ax.axhline(0,color=GRID,lw=0.6); ax.axvline(0,color=GRID,lw=0.6)
ax.set_xlabel('推計の残差（日）'); ax.set_ylabel('実測の残差（日）')
ax.text(0.04,0.96,f'相関 {r1:.2f}（内挿のみ {r0:.2f}）\n傾き {b:.2f}',transform=ax.transAxes,va='top',fontsize=9,color=INK)
ax.set_title('c　年変動の再現（局と県×年度の固定効果を除去）',loc='left',fontsize=10.5)
print('within r',round(r0,3),round(r1,3),round(b,2))
fig.tight_layout(); fig.savefig(OUT+'fig2_validation.png',dpi=300,bbox_inches='tight'); plt.close()
# ---------- 図3 ----------
from decimal import Decimal,ROUND_HALF_UP
R3=lambda x:str(Decimal(str(x)).quantize(Decimal('0.001'),ROUND_HALF_UP))
rows=[('主解析と固定効果',None),
 ('M1 圏域＋年度',(1.0146,1.0089,1.0203)),('M2 ＋年度×人口密度（主解析）',(1.0105,1.0029,1.0182)),
 ('M3 ＋地方×年度',(1.0092,1.0018,1.0166)),('M4 ＋県×年度',(1.0094,1.0016,1.0173)),
 ('2018年度を除外',None),('M2',(1.0051,0.9981,1.0121)),('M4',(1.0089,1.0000,1.0178)),
 ('時間的な特異性（M2）',None),('当年の夏（2018–2022年度）',(1.0137,1.0026,1.0250)),('翌年の夏（2018–2022年度）',(1.0003,0.9938,1.0067)),('前年の夏（2019–2023年度）',(1.0037,0.9969,1.0104)),
 ('日中の暑さ・湿度・大気汚染で調整（M2）',None),('＋平均最高気温',(1.0110,1.0033,1.0187)),('＋猛暑日',(1.0116,1.0049,1.0184)),('＋平均最高気温＋水蒸気圧',(1.0113,1.0038,1.0189)),('＋NO$_2$',(1.0105,1.0029,1.0182)),
 ('モデルと暴露の別仕様',None),('性・年齢別の集計二項モデル',(1.0109,1.0036,1.0182)),('都市補正なしの暴露',(1.0096,1.0023,1.0170))]
fig,ax=plt.subplots(figsize=(8,8.4)); y=0; yt=[]; yl=[]
for lab,v in rows[::-1]:
    if v is None: ax.text(0.975,y,lab,fontsize=9.5,va='center',gid='hd',color=INK,transform=ax.get_yaxis_transform()); 
    else:
        main='主解析' in lab; c=C2 if main else C1
        ax.plot([v[1],v[2]],[y,y],color=c,lw=2); ax.plot(v[0],y,'o',color=c,ms=7 if main else 6,mec='white',mew=1)
        ax.text(1.032,y,f'{R3(v[0])}（{R3(v[1])}–{R3(v[2])}）',fontsize=8.5,va='center',color=INK)
        yt.append(y); yl.append(lab)
    y+=1
ax.axvline(1,color=GRID,lw=1); ax.set_yticks(yt,yl,fontsize=9); ax.set_xlim(0.985,1.03); ax.set_ylim(-0.8,y-0.2)
ax.set_xlabel('睡眠休養不足のオッズ比（熱帯夜10日あたり、95% CI）'); ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
# 見出しは左端に
for t in ax.texts:
    if t.get_gid()=='hd': t.set_x(-0.62)
fig.savefig(OUT+'fig3_forest.png',dpi=300,bbox_inches='tight'); plt.close()
# ---------- 表1 ----------
T=[]
for yr,d in D.groupby('year'):
    T.append(dict(年度=yr,圏域数=d.area.nunique(),回答数=int(d.tot.sum()),熱帯夜_全国平均=round(np.average(d.tn25,weights=d.tot),1),
      熱帯夜_圏域5パーセンタイル=round(d.tn25.quantile(.05),1),熱帯夜_圏域95パーセンタイル=round(d.tn25.quantile(.95),1),
      熱帯夜が1日以上の圏域=int((d.tn25>=1).sum()),休養不足_標準化割合_全国=round(100*np.average(d.p,weights=d.tot),1),
      休養不足_圏域5パーセンタイル=round(100*d.p.quantile(.05),1),休養不足_圏域95パーセンタイル=round(100*d.p.quantile(.95),1)))
T=pd.DataFrame(T); print(T.to_string()); T.to_csv(OUT+'table1_by_year.csv',index=False,encoding='utf-8-sig')
