exec(open('hn/panel_v5.py').read().split("OUT=[]")[0].replace("D=D.merge(pd.concat(UL)","D0=None\nD=D.merge(pd.concat(UL)"))
OUT=[]
for lab,fes in [('M2',['area','yr']),('M4',['area','py'])]:
    for nm,d in [('沖縄除外',D[D.prefc!='47']),('沖縄除外・2018除外',D[(D.prefc!='47')&(D.year!=2018)]),('東北除外',D[D.region!='TH']),('東北・沖縄除外 2018除外',D[(D.region!='TH')&(D.prefc!='47')&(D.year!=2018)])]:
        fe(d,['tn10u'],fes,YD,f'{lab} {nm}')
