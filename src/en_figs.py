# English versions of Figures 1-4 and Supplementary Figure 1 (run from scratchpad root)
import pandas as pd,numpy as np,geopandas as gpd,statsmodels.api as sm,matplotlib,sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.lines as mlines
from matplotlib import rcParams; rcParams['font.family']='DejaVu Sans'; rcParams['axes.spines.top']=False; rcParams['axes.spines.right']=False
OUT='/mnt/project-files/paper_E_hotnight/figures/en/'
INK='#333333'; C1='#2a78d6'; C2='#eb6834'; GRID='#bbbbbb'; MUTE='#666666'
GPKG='/mnt/project-files/paper_B_analysis_2026-11-09/figures/areas_335_simplified.gpkg'
def save(fig,name):
    fig.savefig(OUT+name+'.png',dpi=300,bbox_inches='tight'); fig.savefig(OUT+name+'.pdf',bbox_inches='tight'); plt.close(fig)
def okinawa(g):
    ok=g.code.str[:2]=='47'; g.loc[ok,'geometry']=g.loc[ok,'geometry'].translate(xoff=5.0,yoff=13.0); return g
def frame(ax):
    ax.plot([128.5,136.6,136.6],[36.6,36.6,40.6],color=GRID,lw=0.8); ax.text(128.7,40.2,'Okinawa',fontsize=8,color=INK)
    ax.set_xlim(128.3,146.2); ax.set_ylim(30.0,45.8); ax.set_axis_off()
D=pd.read_pickle('hn/panel_v5_D.pkl'); D['tn25']=D.tn10u*10
which=sys.argv[1:] or ['1','2','3','4','s']
if '1' in which:
    A=pd.read_csv('hn/area_hotnight_uhi.csv',dtype={'code':str}); g=okinawa(gpd.read_file(GPKG).merge(A,on='code'))
    fig=plt.figure(figsize=(12,5.2)); gs=fig.add_gridspec(1,3,width_ratios=[1,1,0.85],wspace=0.08); vmax=60; cmap=plt.get_cmap('OrRd')
    for i,yr in enumerate([2021,2023]):
        ax=fig.add_subplot(gs[0,i]); g.plot(column=f'tn25_u_{yr}',ax=ax,cmap=cmap,vmin=0,vmax=vmax,edgecolor='white',linewidth=0.15)
        frame(ax); ax.set_title(f'{"ab"[i]}   {yr}, July–September',loc='left',fontsize=11)
    sm_=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,vmax)); cax=fig.add_axes([0.14,0.08,0.4,0.025])
    cb=fig.colorbar(sm_,cax=cax,orientation='horizontal',extend='max'); cb.set_label('Tropical nights (days; population-weighted by area, land-use corrected)',fontsize=9)
    ax=fig.add_subplot(gs[0,2]); ax.set_position([0.70,0.2,0.27,0.62])
    m=D.groupby('year').apply(lambda d:np.average(d.tn25,weights=d.tot)); print(m.round(1).to_dict())
    ax.bar(m.index,m.values,color=C2,width=0.6)
    for x,v in m.items(): ax.text(x,v+0.6,f'{v:.1f}',ha='center',fontsize=8.5,color=INK)
    ax.set_xticks(m.index,[f'{y}' for y in m.index],fontsize=8.5); ax.set_ylabel('Tropical nights (days, national mean)'); ax.set_ylim(0,38)
    ax.set_title('c   National mean by year',loc='left',fontsize=11)
    ax.text(0,-0.2,'Area values weighted by the number of checkup responses',transform=ax.transAxes,fontsize=8,color=MUTE)
    save(fig,'fig1_tropical_nights_map')
if '2' in which:
    P=pd.read_csv('hn/uhi_station_pred_d0.2.csv'); P['kind']=P.kind.astype(str)
    fig,axs=plt.subplots(1,3,figsize=(13,4.6))
    def lvl(ax,col,t):
        for k,c,lab in [('1',C1,'General stations'),('2',C2,'Roadside stations')]:
            s=P[P.kind==k]; ax.scatter(s[col],s.tn25,s=8,c=c,alpha=0.45,edgecolors='none',label=lab)
        s=P[~P.kind.isin(['1','2'])]; ax.scatter(s[col],s.tn25,s=8,c='#999999',alpha=0.45,edgecolors='none',label='Other')
        e=P[col]-P.tn25; r2=1-(e**2).sum()/((P.tn25-P.tn25.mean())**2).sum()
        ax.plot([0,100],[0,100],color=INK,lw=1); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect('equal')
        ax.set_xlabel('Estimated tropical nights (days)'); ax.set_ylabel('Observed tropical nights at station (days)')
        ax.text(0.04,0.96,f'R² {r2:.2f}\nBias {e.mean():+.1f} days\nRMSE {np.sqrt((e**2).mean()):.1f} days'.replace('-','−'),transform=ax.transAxes,va='top',fontsize=9,color=INK)
        ax.set_title(t,loc='left',fontsize=10.5)
    lvl(axs[0],'p_idw','a   Interpolation only'); lvl(axs[1],'p_final_noc','b   Land-use correction (main)')
    axs[0].legend(loc='lower right',frameon=False,fontsize=8.5,markerscale=2)
    d=P.copy(); d['py']=d.pref.astype(str)+'_'+d.fy.astype(str)
    for c in ['tn25','p_idw','p_final_noc']:
        d[c+'r']=d[c].astype(float)
        for it in range(100):
            for gcol in ['code','py']: d[c+'r']-=d.groupby(gcol)[c+'r'].transform('mean')
    ax=axs[2]; ax.scatter(d.p_final_nocr,d.tn25r,s=8,c=C1,alpha=0.45,edgecolors='none')
    r0=np.corrcoef(d.p_idwr,d.tn25r)[0,1]; r1=np.corrcoef(d.p_final_nocr,d.tn25r)[0,1]; b=np.polyfit(d.p_final_nocr,d.tn25r,1)[0]
    lim=25; ax.plot([-lim,lim],[-lim,lim],color=INK,lw=1); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
    ax.axhline(0,color=GRID,lw=0.6); ax.axvline(0,color=GRID,lw=0.6)
    ax.set_xlabel('Estimated, residual (days)'); ax.set_ylabel('Observed, residual (days)')
    ax.text(0.04,0.96,f'r = {r1:.2f} (interpolation only {r0:.2f})\nSlope {b:.2f}',transform=ax.transAxes,va='top',fontsize=9,color=INK)
    ax.set_title('c   Year-to-year changes\n     (station and prefecture × year FE removed)',loc='left',fontsize=10.5)
    print('within r',round(r0,3),round(r1,3),round(b,2))
    fig.tight_layout(); save(fig,'fig2_validation')
if '3' in which:
    from decimal import Decimal,ROUND_HALF_UP
    R3=lambda x:str(Decimal(str(x)).quantize(Decimal('0.001'),ROUND_HALF_UP))
    rows=[('Main model and fixed effects',None),
     ('M1 Area + year',(1.0146,1.0089,1.0203)),('M2 + year × population density (main)',(1.0105,1.0029,1.0182)),
     ('M3 Year → region × year',(1.0092,1.0018,1.0166)),('M4 Year → prefecture × year',(1.0094,1.0016,1.0173)),
     ('Without 2018',None),('M2',(1.0051,0.9981,1.0121)),('M4',(1.0089,1.0000,1.0178)),
     ('Temporal specificity (M2)',None),('Same summer (2018–2022)',(1.0137,1.0026,1.0250)),('Next summer (2018–2022)',(1.0003,0.9938,1.0067)),('Previous summer (2019–2023)',(1.0037,0.9969,1.0104)),
     ('Adjustment for daytime heat, humidity, air pollution (M2)',None),('+ mean daily maximum temperature',(1.0110,1.0033,1.0187)),('+ days with maximum ≥35 °C',(1.0116,1.0049,1.0184)),('+ mean daily maximum temperature + vapor pressure',(1.0113,1.0038,1.0189)),('+ NO$_2$',(1.0105,1.0029,1.0182)),
     ('Other model and exposure specifications',None),('Aggregated binomial model by sex and age',(1.0109,1.0036,1.0182)),('Exposure without land-use correction',(1.0096,1.0023,1.0170))]
    fig,ax=plt.subplots(figsize=(8.6,8.4)); y=0; yt=[]; yl=[]
    for lab,v in rows[::-1]:
        if v is None: ax.text(0.975,y,lab,fontsize=9.5,va='center',gid='hd',color=INK,weight='bold')
        else:
            main='(main)' in lab; c=C2 if main else C1
            ax.plot([v[1],v[2]],[y,y],color=c,lw=2); ax.plot(v[0],y,'o',color=c,ms=7 if main else 6,mec='white',mew=1)
            ax.text(1.032,y,f'{R3(v[0])} ({R3(v[1])}–{R3(v[2])})',fontsize=8.5,va='center',color=INK)
            yt.append(y); yl.append(lab)
        y+=1
    ax.axvline(1,color=GRID,lw=1); ax.set_yticks(yt,yl,fontsize=9); ax.set_xlim(0.985,1.03); ax.set_ylim(-0.8,y-0.2)
    ax.set_xlabel('Odds ratio of insufficient rest from sleep\nper 10 tropical nights (95% CI)'); ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
    fig.canvas.draw(); x0=min(t.get_window_extent().x0 for t in ax.get_yticklabels()); xa=(x0-ax.bbox.x0)/ax.bbox.width
    for t in ax.texts:
        if t.get_gid()=='hd': t.set_transform(matplotlib.transforms.blended_transform_factory(ax.transAxes,ax.transData)); t.set_x(xa)
    save(fig,'fig3_forest')
if '4' in which:
    exec(open('hn/v6.py').read().split("# --- 図")[0]); rcParams['font.family']='DejaVu Sans'
    fig,axs=plt.subplots(1,2,figsize=(10,4.9),sharex=True,sharey=True)
    for ax,(lab,fes) in zip(axs,[('M2  Area + year + year × population density',['area','yr']),('M4  Area + prefecture × year + year × population density',['area','py'])]):
        P=partial(D,fes); P['year']=D.year.values; P['w']=D.tot.values
        xs=P.tn10u*10; ys=P.y*100; s=8+80*P.w/P.w.max()
        for is18,c,z in [(False,C1,2),(True,C2,3)]:
            m=(P.year==2018)==is18; ax.scatter(xs[m],ys[m],s=s[m],c=c,alpha=0.55,edgecolors='white',linewidths=0.5,zorder=z)
        xx=np.linspace(-13,15,10); b=sm.WLS(P.y*100,P.tn10u*10,weights=P.w).fit().params.iloc[0]
        ax.plot(xx,b*xx,'-',color=INK,lw=2,zorder=4)
        ax.text(0.02,0.04,f'All-year regression line: OR {np.exp(b/100*10):.3f} per 10 nights',transform=ax.transAxes,fontsize=8.5,color=INK,va='bottom')
        ax.axhline(0,color=GRID,lw=0.8,zorder=1); ax.axvline(0,color=GRID,lw=0.8,zorder=1)
        ax.set_title(lab,fontsize=10); ax.set_xlabel('Adjusted tropical nights (days)')
    axs[0].set_ylabel('Adjusted residual of insufficient rest from sleep\n(logit scale × 100)'); axs[0].set_xlim(-13,15); axs[0].set_ylim(-50,25)
    h=[mlines.Line2D([],[],marker='o',ls='',color=C1,alpha=.7,label='2019–2023'),mlines.Line2D([],[],marker='o',ls='',color=C2,alpha=.7,label='2018'),
       mlines.Line2D([],[],color=INK,lw=2,label='All-year regression line')]
    fig.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=9)
    plt.tight_layout(rect=(0,0.07,1,1)); save(fig,'fig4_scatter_2018')
if 's' in which:
    E=pd.read_csv('/mnt/project-files/paper_E_hotnight/figures/suppfig_excess_by_area_2023.csv',dtype={'code':str})
    G=okinawa(gpd.read_file(GPKG).merge(E,on='code'))
    fig,axs=plt.subplots(1,2,figsize=(11,5.8))
    for ax,col,vmin,vmax,lab,t in [(axs[0],'dtn',-30,30,'2023 minus the 2018–2022 mean (days)','a   Change in tropical nights'),
                                   (axs[1],'per1000',-8,8,'Per 1,000 checkup respondents','b   Modeled difference in insufficient rest from sleep')]:
        G.plot(column=col,ax=ax,cmap='RdBu_r',vmin=vmin,vmax=vmax,edgecolor='white',linewidth=0.15,legend=True,
               legend_kwds={'orientation':'horizontal','shrink':0.7,'pad':0.02,'label':lab,'extend':'both'})
        frame(ax); ax.set_title(t,loc='left',fontsize=11)
    save(fig,'suppfig1_excess_map')
