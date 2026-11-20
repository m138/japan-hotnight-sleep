import pandas as pd, geopandas as gpd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.family']='DejaVu Sans'
E=pd.read_csv('/mnt/project-files/paper_E_hotnight/figures/suppfig_excess_by_area_2023.csv',dtype={'code':str},encoding='utf-8-sig')
E['code']=E.code.str.zfill(4)
G=gpd.read_file('/mnt/project-files/paper_B_analysis_2026-11-09/figures/areas_335_simplified.gpkg').merge(E,on='code')
ok=G.code.str[:2]=='47'; G.loc[ok,'geometry']=G.loc[ok,'geometry'].translate(xoff=4.5,yoff=14.0)
ink='#1f2328'; mute='#57606a'; line='#d0d7de'
fig=plt.figure(figsize=(13.28,5.31),dpi=200); fig.patch.set_facecolor('white')
fig.text(0.015,0.93,'Tropical nights and insufficient rest from sleep in Japan',fontsize=19,weight='bold',color=ink)
fig.text(0.015,0.865,'335 secondary medical care areas  ·  fiscal years 2018–2023  ·  159.3 million health checkup responses (ages 40–74)',fontsize=12,color=mute)
# panel 1 map
ax=fig.add_axes([0.0,0.15,0.33,0.62])
G.plot(column='dtn',ax=ax,cmap='RdBu_r',vmin=-30,vmax=30,edgecolor='white',linewidth=0.1)
ax.plot([127.0,136.0,136.0],[37.7,37.7,42.2],color='#bbbbbb',lw=0.8)
ax.set_xlim(126.8,146.2); ax.set_ylim(30.0,45.8); ax.set_axis_off()
sm=plt.cm.ScalarMappable(cmap='RdBu_r',norm=plt.Normalize(-30,30)); cax=fig.add_axes([0.06,0.10,0.22,0.025])
cb=fig.colorbar(sm,cax=cax,orientation='horizontal',extend='both'); cb.ax.tick_params(labelsize=9,colors=mute); cb.outline.set_visible(False)
cb.set_label('Population-weighted days with daily minimum ≥25 °C (Jul–Sep)',fontsize=9,color=mute)
fig.text(0.02,0.76,'Change in tropical nights',fontsize=13,weight='bold',color=ink)
fig.text(0.02,0.715,'2023 minus the 2018–2022 mean',fontsize=10.5,color=mute)
def box(x,w,title,lines,gap=0.022):
    fig.patches.append(FancyBboxPatch((x,0.06),w,0.70,boxstyle='round,pad=0.005,rounding_size=0.015',transform=fig.transFigure,fc='#f6f8fa',ec=line,lw=1))
    fig.text(x+0.02,0.69,title,fontsize=13,weight='bold',color=ink)
    y=0.64
    for s,size,col,wt in lines:
        y-=size*0.35/382; fig.text(x+0.02,y,s,fontsize=size,color=col,weight=wt,va='top'); y-=size*1.0/382+gap
box(0.35,0.27,'Validation',[('376 air quality monitoring stations',12,ink,'normal'),('leave-one-prefecture-out cross-validation',11,mute,'normal'),('RMSE of tropical nights',12,ink,'normal'),('9.3 → 8.3 days',22,ink,'bold'),('with land-use correction',11,mute,'normal'),('Within-station correlation: 0.53 → 0.57',12,ink,'normal'),('after adjustment',11,mute,'normal')])
box(0.635,0.355,'Within-area association',[('10 more tropical nights in a year',12,ink,'normal'),('OR 1.011',26,'#b2182b','bold'),('(95% CI 1.003–1.018)',11,mute,'normal'),('With prefecture-by-year fixed effects: OR 1.009',11,ink,'normal'),('2023 respondents under two exposure scenarios',11.5,ink,'bold'),('2023 tropical-night estimates vs each area’s 2018–2022 mean',9.5,mute,'normal'),('3 more predicted reports per 1,000 respondents',11,ink,'normal'),('≈85,000 additional reports nationwide',11,ink,'normal'),('among checkup respondents',11,mute,'normal')],gap=0.012)
fig.savefig('/mnt/project-files/paper_E_hotnight/figures/en/graphical_abstract.png',dpi=200,facecolor='white')
fig.savefig('/mnt/project-files/paper_E_hotnight/figures/en/graphical_abstract.pdf',facecolor='white')
