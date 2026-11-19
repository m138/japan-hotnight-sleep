exec(open('hn/v6.py').read().split("# --- 図")[0])
import matplotlib.lines as mlines
fig,axs=plt.subplots(1,2,figsize=(10,4.9),sharex=True,sharey=True)
C_OTHER,C_2018='#2a78d6','#eb6834'
for ax,(lab,fes) in zip(axs,[('M2 圏域＋年度＋年度×人口密度',['area','yr']),('M4 圏域＋県×年度＋年度×人口密度',['area','py'])]):
    P=partial(D,fes); P['year']=D.year.values; P['w']=D.tot.values
    xs=P.tn10u*10; ys=P.y*100; s=8+80*P.w/P.w.max()
    for is18,c,z in [(False,C_OTHER,2),(True,C_2018,3)]:
        m=(P.year==2018)==is18
        ax.scatter(xs[m],ys[m],s=s[m],c=c,alpha=0.55,edgecolors='white',linewidths=0.5,zorder=z)
    xx=np.linspace(-13,15,10); txt=[]
    for sub,ls,nm in [(P,'-','全年度の回帰線')]:
        b=sm.WLS(sub.y*100,sub.tn10u*10,weights=sub.w).fit().params.iloc[0]
        ax.plot(xx,b*xx,ls,color='#333333',lw=2,zorder=4); txt.append(f'{nm}：OR {np.exp(b/100*10):.3f}/10夜')
    ax.text(0.02,0.04,'\n'.join(txt),transform=ax.transAxes,fontsize=8.5,color='#333333',va='bottom')
    ax.axhline(0,color='#bbbbbb',lw=0.8,zorder=1); ax.axvline(0,color='#bbbbbb',lw=0.8,zorder=1)
    ax.set_title(lab,fontsize=10.5); ax.set_xlabel('調整後の熱帯夜日数（日）')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
axs[0].set_ylabel('睡眠休養不足割合の調整後残差\n（ロジット尺度×100）')
axs[0].set_xlim(-13,15); axs[0].set_ylim(-50,25)
h=[mlines.Line2D([],[],marker='o',ls='',color=C_OTHER,alpha=.7,label='2019–2023年度'),mlines.Line2D([],[],marker='o',ls='',color=C_2018,alpha=.7,label='2018年度'),
   mlines.Line2D([],[],color='#333333',lw=2,label='全年度の回帰線')]
fig.legend(handles=h,loc='lower center',ncol=3,frameon=False,fontsize=9)
plt.tight_layout(rect=(0,0.07,1,1)); plt.savefig('/mnt/project-files/paper_E_hotnight/figures/fig4_scatter_2018.png',dpi=200)
